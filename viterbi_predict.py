import os
import glob
import commentjson as json
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

# ----------------------------
# Config
# ----------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Defaults (safe)
    cfg.setdefault("search", {})
    cfg["search"].setdefault("max_fret", 20)
    cfg["search"].setdefault("per_pitch_k", 4)
    cfg["search"].setdefault("chord_k", 40)
    cfg["search"].setdefault("beam_size", 30)

    cfg.setdefault("local_cost", {})
    cfg["local_cost"].setdefault("w_span", 1.2)
    cfg["local_cost"].setdefault("w_high", 0.25)
    cfg["local_cost"].setdefault("high_fret_threshold", 12)
    cfg["local_cost"].setdefault("w_open_bonus", -0.05)
    cfg["local_cost"].setdefault("w_string_range", 0.10)
    cfg["local_cost"].setdefault("preferred_min_fret", 0)
    cfg["local_cost"].setdefault("preferred_max_fret", 24)
    cfg["local_cost"].setdefault("w_preferred_zone", 0.0)  # negative = reward
    cfg["local_cost"].setdefault("high_string_threshold", 1)  # strings <= threshold are "high"
    cfg["local_cost"].setdefault("w_high_string", 0.0)  # penalty per note on a high string

    cfg.setdefault("string_discontinuity", {})
    cfg["string_discontinuity"].setdefault("w_holes", 1.8)
    cfg["string_discontinuity"].setdefault("w_gap", 0.8)
    cfg["string_discontinuity"].setdefault("w_blocks", 1.2)

    cfg.setdefault("transition_cost", {})
    tc = cfg["transition_cost"]
    tc.setdefault("w_jump", 1.4)
    tc.setdefault("jump_power", 1.7)
    tc.setdefault("jump_threshold", 5.0)
    tc.setdefault("jump_threshold_penalty", 6.0)
    tc.setdefault("w_avg_jump", 0.6)
    tc.setdefault("avg_jump_power", 1.3)
    tc.setdefault("w_span_change", 0.25)
    tc.setdefault("w_string_center", 0.5)
    tc.setdefault("close_jump_threshold", 2.0)
    tc.setdefault("close_jump_bonus", -1.2)
    # optional rest handling
    tc.setdefault("rest_enter_penalty", 0.0)
    tc.setdefault("rest_exit_penalty", 0.0)
    # directional streak penalty (4-finger limit)
    tc.setdefault("w_streak", 2.0)
    tc.setdefault("streak_min_len", 4)
    tc.setdefault("streak_speed_threshold", 480)  # ticks; 480 = 8th note at 960tpq

    cfg.setdefault("guitar", {})
    # E standard 6-string: string 1 (high E) to string 6 (low E)
    cfg["guitar"].setdefault("tuning", [64, 59, 55, 50, 45, 40])

    cfg.setdefault("io", {})
    cfg["io"].setdefault("jsonl_dir", "./test_input_folder")
    cfg["io"].setdefault("out_pred_dir", "./test_output_folder")
    cfg["io"].setdefault("limit_files", 50)

    return cfg


# ----------------------------
# Instrument model
# ----------------------------
def tuning_dict(cfg: Dict[str, Any]) -> Dict[int, int]:
    """Returns {string_number: open_midi_pitch} from config tuning list."""
    return {i + 1: int(p) for i, p in enumerate(cfg["guitar"]["tuning"])}

def pitch_to_positions(pitch: int, tuning: Dict[int, int], max_fret: int = 20) -> List[Tuple[int, int]]:
    out = []
    for s, open_pitch in tuning.items():
        fret = pitch - open_pitch
        if 0 <= fret <= max_fret:
            out.append((s, fret))
    return out

def chord_span_ignore_open(frets: List[int]) -> int:
    fretted = [f for f in frets if f > 0]
    if not fretted:
        return 0
    return max(fretted) - min(fretted)

def anchor_pos(frets: List[int]) -> float:
    fretted = [f for f in frets if f > 0]
    return float(min(fretted)) if fretted else 0.0

def avg_fretted_pos(frets: List[int]) -> float:
    fretted = [f for f in frets if f > 0]
    if not fretted:
        return 0.0
    return float(sum(fretted) / len(fretted))

# ----------------------------
# Data structures
# ----------------------------
@dataclass(frozen=True)
class Voicing:
    pitches: Tuple[int, ...]
    strings: Tuple[int, ...]
    frets: Tuple[int, ...]
    local_cost: float
    span: int
    anchor: float
    avgpos: float

# ----------------------------
# Candidate generation
# ----------------------------
def per_pitch_candidates(pitches: List[int], tuning: Dict[int, int], max_fret: int, per_pitch_k: int) -> List[List[Tuple[int,int]]]:
    per = []
    for p in pitches:
        poss = pitch_to_positions(p, tuning, max_fret=max_fret)
        if not poss:
            per.append([])
            continue

        def note_score(sf):
            s, f = sf
            return (
                f,
                0 if f == 0 else 1,
                abs(s - 3.5),
            )

        poss_sorted = sorted(poss, key=note_score)
        per.append(poss_sorted[:per_pitch_k])
    return per


def generate_voicings_for_event(
    pitches: List[int],
    cfg: Dict[str, Any],
) -> List[Voicing]:
    """
    Generate top chord_k voicings for this event.
    Constraint: no two notes on the same string.
    """
    search = cfg["search"]
    max_fret = int(search["max_fret"])
    per_pitch_k = int(search["per_pitch_k"])
    chord_k = int(search["chord_k"])
    tuning = tuning_dict(cfg)

    # REST event
    if not pitches:
        return [Voicing(
            pitches=tuple(),
            strings=tuple(),
            frets=tuple(),
            local_cost=0.0,
            span=0,
            anchor=0.0,
            avgpos=0.0,
        )]

    pitches_sorted = sorted([int(p) for p in pitches])
    per = per_pitch_candidates(pitches_sorted, tuning, max_fret=max_fret, per_pitch_k=per_pitch_k)
    if any(len(cands) == 0 for cands in per):
        return []

    best: List[Voicing] = []

    used_strings = set()
    chosen_strings: List[int] = []
    chosen_frets: List[int] = []

    disc = cfg["string_discontinuity"]
    lc_cfg = cfg["local_cost"]

    def string_discontinuity_penalty(strings: List[int]) -> float:
        if not strings:
            return 0.0
        s = sorted(strings)
        span = (s[-1] - s[0] + 1)
        holes = span - len(s)
        max_gap = max(b - a for a, b in zip(s, s[1:])) if len(s) > 1 else 0

        blocks = 1
        for a, b in zip(s, s[1:]):
            if b != a + 1:
                blocks += 1

        w_holes = float(disc["w_holes"])
        w_gap = float(disc["w_gap"])
        w_blocks = float(disc["w_blocks"])

        return (w_holes * holes) + (w_gap * max(0, max_gap - 1)) + (w_blocks * max(0, blocks - 1))

    def local_cost_for_assignment(strings: List[int], frets: List[int]) -> float:
        span = chord_span_ignore_open(frets)

        w_span = float(lc_cfg["w_span"])
        w_high = float(lc_cfg["w_high"])
        high_thr = int(lc_cfg["high_fret_threshold"])
        w_open_bonus = float(lc_cfg["w_open_bonus"])
        w_string_range = float(lc_cfg["w_string_range"])
        pref_min = int(lc_cfg["preferred_min_fret"])
        pref_max = int(lc_cfg["preferred_max_fret"])
        w_pref = float(lc_cfg["w_preferred_zone"])
        high_str_thr = int(lc_cfg["high_string_threshold"])
        w_high_str = float(lc_cfg["w_high_string"])

        high_pen = sum(max(0, f - high_thr) for f in frets)
        open_bonus = sum(1 for f in frets if f == 0)
        in_zone = sum(1 for f in frets if f > 0 and pref_min <= f <= pref_max)
        high_str_pen = sum(1 for s in strings if s <= high_str_thr)

        cost = (w_span * span) + (w_high * high_pen) + (w_open_bonus * open_bonus) + (w_pref * in_zone) + (w_high_str * high_str_pen)

        if strings:
            cost += w_string_range * (max(strings) - min(strings))

        cost += string_discontinuity_penalty(strings)

        return float(cost)

    def push_best(v: Voicing):
        best.append(v)
        best.sort(key=lambda x: x.local_cost)
        if len(best) > chord_k:
            best.pop()

    def backtrack(i: int):
        if i == len(pitches_sorted):
            frets = chosen_frets[:]
            strings = chosen_strings[:]
            lc = local_cost_for_assignment(strings, frets)
            v = Voicing(
                pitches=tuple(pitches_sorted),
                strings=tuple(strings),
                frets=tuple(frets),
                local_cost=lc,
                span=chord_span_ignore_open(frets),
                anchor=anchor_pos(frets),
                avgpos=avg_fretted_pos(frets),
            )
            push_best(v)
            return

        if len(best) >= chord_k:
            worst = best[-1].local_cost
            partial = local_cost_for_assignment(chosen_strings, chosen_frets) if chosen_frets else 0.0
            if partial > worst:
                return

        for (s, f) in per[i]:
            if s in used_strings:
                continue
            used_strings.add(s)
            chosen_strings.append(s)
            chosen_frets.append(f)
            backtrack(i + 1)
            chosen_frets.pop()
            chosen_strings.pop()
            used_strings.remove(s)

    backtrack(0)
    return sorted(best, key=lambda x: x.local_cost)


# ----------------------------
# Viterbi (beam search)
# ----------------------------
@dataclass
class BeamNode:
    score: float
    voicing: Voicing
    prev_index: Optional[int]
    streak_dir: int = 0   # -1 = going down, 0 = no streak, +1 = going up
    streak_len: int = 0   # number of consecutive fast single-note steps in streak_dir

def transition_cost(prev: Voicing, cur: Voicing, cfg: Dict[str, Any]) -> float:
    tc = cfg["transition_cost"]

    jump = abs(cur.anchor - prev.anchor)
    avg_jump = abs(cur.avgpos - prev.avgpos)
    span_change = abs(cur.span - prev.span)

    cost = 0.0

    # optional: treat entering/leaving rest
    if len(prev.pitches) == 0 and len(cur.pitches) > 0:
        cost += float(tc["rest_enter_penalty"])
    if len(prev.pitches) > 0 and len(cur.pitches) == 0:
        cost += float(tc["rest_exit_penalty"])

    w_jump = float(tc["w_jump"])
    jump_pow = float(tc["jump_power"])
    cost += w_jump * (jump ** jump_pow)

    if jump > float(tc["jump_threshold"]):
        cost += float(tc["jump_threshold_penalty"])

    w_avg = float(tc["w_avg_jump"])
    avg_pow = float(tc["avg_jump_power"])
    cost += w_avg * (avg_jump ** avg_pow)

    cost += float(tc["w_span_change"]) * span_change

    prev_center = (sum(prev.strings) / len(prev.strings)) if prev.strings else 0.0
    cur_center = (sum(cur.strings) / len(cur.strings)) if cur.strings else 0.0
    cost += float(tc["w_string_center"]) * abs(prev_center - cur_center)

    if jump <= float(tc["close_jump_threshold"]):
        cost += float(tc["close_jump_bonus"])  # negative => bonus

    return float(cost)


def compute_streak(
    prev_node: "BeamNode",
    cur_v: Voicing,
    cur_dur: float,
    cfg: Dict[str, Any],
) -> Tuple[int, int, float]:
    """
    Returns (new_streak_dir, new_streak_len, streak_penalty).

    A streak is a run of consecutive fast single-note events where the anchor
    position moves in the same direction. This models the 4-finger limit:
    crawling up or down the neck more than 4 steps in a row at speed is
    physically awkward. Chords, rests, or slow notes reset the streak.
    """
    tc = cfg["transition_cost"]
    speed_threshold = float(tc["streak_speed_threshold"])
    streak_min_len = int(tc["streak_min_len"])
    w_streak = float(tc["w_streak"])

    is_fast = cur_dur <= speed_threshold
    is_single = len(cur_v.pitches) == 1

    # Chord, rest, or slow note: reset streak, no penalty
    if not is_fast or not is_single:
        return 0, 0, 0.0

    delta = cur_v.anchor - prev_node.voicing.anchor
    if delta > 0:
        cur_dir = 1
    elif delta < 0:
        cur_dir = -1
    else:
        # No movement: reset streak
        return 0, 0, 0.0

    if prev_node.streak_dir == cur_dir and prev_node.streak_len > 0:
        new_len = prev_node.streak_len + 1
    else:
        new_len = 1

    penalty = w_streak * max(0, new_len - streak_min_len)
    return cur_dir, new_len, penalty


def viterbi_decode(events: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Voicing]:
    search = cfg["search"]
    beam_size = int(search["beam_size"])

    candidates: List[List[Voicing]] = []
    for ev in events:
        pitches = ev.get("pitches") or [n["pitch"] for n in ev.get("notes", [])]
        pitches = [int(p) for p in pitches]
        voicings = generate_voicings_for_event(pitches, cfg)
        candidates.append(voicings)

    beam: List[BeamNode] = [BeamNode(score=v.local_cost, voicing=v, prev_index=None, streak_dir=0, streak_len=0) for v in candidates[0]]
    beam.sort(key=lambda n: n.score)
    beam = beam[:beam_size]

    backpointers: List[List[Optional[int]]] = [[n.prev_index for n in beam]]
    beams: List[List[BeamNode]] = [beam]

    for t in range(1, len(events)):
        cur_cands = candidates[t]
        new_beam: List[BeamNode] = []

        if not cur_cands:
            new_beam = [BeamNode(score=n.score, voicing=n.voicing, prev_index=i, streak_dir=n.streak_dir, streak_len=n.streak_len) for i, n in enumerate(beam)]
            new_beam.sort(key=lambda n: n.score)
            new_beam = new_beam[:beam_size]
            beam = new_beam
            beams.append(beam)
            backpointers.append([n.prev_index for n in beam])
            continue

        cur_dur = float(events[t].get("dur", 960))

        for cur_v in cur_cands:
            best_score = float("inf")
            best_prev = None
            best_streak_dir = 0
            best_streak_len = 0
            for i, prev_node in enumerate(beam):
                new_dir, new_len, streak_pen = compute_streak(prev_node, cur_v, cur_dur, cfg)
                sc = prev_node.score + transition_cost(prev_node.voicing, cur_v, cfg) + cur_v.local_cost + streak_pen
                if sc < best_score:
                    best_score = sc
                    best_prev = i
                    best_streak_dir = new_dir
                    best_streak_len = new_len
            new_beam.append(BeamNode(score=best_score, voicing=cur_v, prev_index=best_prev, streak_dir=best_streak_dir, streak_len=best_streak_len))

        new_beam.sort(key=lambda n: n.score)
        new_beam = new_beam[:beam_size]
        beam = new_beam
        beams.append(beam)
        backpointers.append([n.prev_index for n in beam])

    if not beam:
        return []

    best_last_idx = min(range(len(beam)), key=lambda i: beam[i].score)
    seq: List[Voicing] = [beam[best_last_idx].voicing]

    idx = best_last_idx
    for t in range(len(events) - 1, 0, -1):
        idx = backpointers[t][idx]
        if idx is None:
            break
        seq.append(beams[t - 1][idx].voicing)

    seq.reverse()
    return seq


# ----------------------------
# Metrics vs ground truth
# ----------------------------
def event_truth_set(ev: Dict[str, Any]) -> set:
    notes = ev.get("notes", [])
    return set((int(n["pitch"]), int(n["string"]), int(n["fret"])) for n in notes)

def event_pred_set(pred: Voicing) -> set:
    return set((int(p), int(s), int(f)) for p, s, f in zip(pred.pitches, pred.strings, pred.frets))

def evaluate_piece(events: List[Dict[str, Any]], preds: List[Voicing]) -> Dict[str, float]:
    assert len(events) == len(preds), "events/preds length mismatch"

    total_notes = 0
    matched_notes = 0
    chord_exact = 0

    pred_spans = []
    pred_jumps = []

    prev_anchor = None
    for ev, pr in zip(events, preds):
        gt = event_truth_set(ev)
        pd = event_pred_set(pr)

        total_notes += len(gt)
        matched_notes += len(gt.intersection(pd))

        if gt == pd:
            chord_exact += 1

        pred_spans.append(float(pr.span))
        if prev_anchor is not None:
            pred_jumps.append(abs(pr.anchor - prev_anchor))
        prev_anchor = pr.anchor

    note_acc = (matched_notes / total_notes) if total_notes > 0 else 0.0
    chord_acc = chord_exact / len(events) if events else 0.0

    pred_spans_sorted = sorted(pred_spans)
    p95_idx = int(0.95 * (len(pred_spans_sorted) - 1)) if len(pred_spans_sorted) > 1 else 0

    return {
        "note_acc": float(note_acc),
        "chord_exact_acc": float(chord_acc),
        "pred_span_mean": float(sum(pred_spans) / len(pred_spans)) if pred_spans else 0.0,
        "pred_span_p95": float(pred_spans_sorted[p95_idx]) if pred_spans else 0.0,
        "pred_jump_mean": float(sum(pred_jumps) / len(pred_jumps)) if pred_jumps else 0.0,
        "pred_jump_max": float(max(pred_jumps)) if pred_jumps else 0.0,
    }


# ----------------------------
# IO helpers
# ----------------------------
def load_events_from_jsonl(path: str) -> List[Dict[str, Any]]:
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    events.sort(key=lambda e: float(e.get("start", 0.0)))
    return events

def save_predicted_jsonl(events: List[Dict[str, Any]], preds: List[Voicing], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ev, pr in zip(events, preds):
            row = {
                "piece_id": ev.get("piece_id"),
                "event_idx": ev.get("event_idx"),
                "start": ev.get("start"),
                "dur": ev.get("dur"),
                "pitches": list(pr.pitches),
                "pred_strings": list(pr.strings),
                "pred_frets": list(pr.frets),
                "pred_span": pr.span,
                "pred_anchor": pr.anchor,
                "pred_avgpos": pr.avgpos,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------
# Main (run on a folder)
# ----------------------------
def run_folder(cfg: Dict[str, Any]) -> None:
    io = cfg["io"]
    jsonl_dir = io["jsonl_dir"]
    out_pred_dir = io["out_pred_dir"]
    limit_files = io.get("limit_files", None)

    files = sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl")))
    if limit_files is not None:
        files = files[:int(limit_files)]
    if not files:
        raise FileNotFoundError(f"No jsonl files found in {jsonl_dir}")

    metrics_all = []
    for fp in files:
        events = load_events_from_jsonl(fp)
        preds = viterbi_decode(events, cfg)
        m = evaluate_piece(events, preds)
        m["file"] = os.path.basename(fp)
        metrics_all.append(m)

        out_path = os.path.join(out_pred_dir, os.path.basename(fp))
        save_predicted_jsonl(events, preds, out_path)

        print(f"[OK] {os.path.basename(fp)}  note_acc={m['note_acc']:.3f}  chord_exact={m['chord_exact_acc']:.3f}  jump_max={m['pred_jump_max']:.1f}  span_p95={m['pred_span_p95']:.1f}")

    note_accs = [x["note_acc"] for x in metrics_all]
    chord_accs = [x["chord_exact_acc"] for x in metrics_all]
    print("\n=== Aggregate ===")
    print(f"Files: {len(metrics_all)}")
    print(f"Mean note_acc:      {sum(note_accs)/len(note_accs):.3f}")
    print(f"Mean chord_exact:   {sum(chord_accs)/len(chord_accs):.3f}")

    metrics_all.sort(key=lambda x: x["note_acc"])
    print("\n=== Worst 10 (note_acc) ===")
    for x in metrics_all[:10]:
        print(f"{x['file']:40s} note_acc={x['note_acc']:.3f} chord_exact={x['chord_exact_acc']:.3f} jump_max={x['pred_jump_max']:.1f} span_p95={x['pred_span_p95']:.1f}")


if __name__ == "__main__":
    # Usage:
    #   python step3_viterbi_baseline.py --config viterbi_config.json
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="viterbi_config.jsonc")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_folder(cfg)
