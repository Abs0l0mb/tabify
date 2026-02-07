import os
import glob
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

# ----------------------------
# Instrument model (E standard)
# ----------------------------
E_STD_OPEN_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

def pitch_to_positions(pitch: int, max_fret: int = 20) -> List[Tuple[int, int]]:
    """
    Returns all (string, fret) positions that can play this pitch in E standard.
    """
    out = []
    for s, open_pitch in E_STD_OPEN_MIDI.items():
        fret = pitch - open_pitch
        if 0 <= fret <= max_fret:
            out.append((s, fret))
    return out

def chord_span_ignore_open(frets: List[int]) -> int:
    """
    Span = max(fretted) - min(fretted), ignoring open strings for min/max.
    If all open or empty -> 0
    """
    fretted = [f for f in frets if f > 0]
    if not fretted:
        return 0
    return max(fretted) - min(fretted)

def anchor_pos(frets: List[int]) -> float:
    """
    Anchor position for a chord/event: min fretted, fallback 0.
    """
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
    """
    One candidate choice for an event: mapping pitches -> (string,fret)
    Stored in pitch-sorted order for determinism.
    """
    pitches: Tuple[int, ...]
    strings: Tuple[int, ...]
    frets: Tuple[int, ...]
    local_cost: float
    span: int
    anchor: float
    avgpos: float

def voicing_set(v: Voicing) -> set:
    return set(zip(v.pitches, v.strings, v.frets))

# ----------------------------
# Candidate generation
# ----------------------------
def per_pitch_candidates(pitches: List[int], max_fret: int, per_pitch_k: int) -> List[List[Tuple[int,int]]]:
    """
    For each pitch, return up to per_pitch_k best (string,fret) options by a heuristic.
    """
    per = []
    for p in pitches:
        poss = pitch_to_positions(p, max_fret=max_fret)
        if not poss:
            per.append([])
            continue

        # Heuristic ranking for single note:
        # - prefer lower frets
        # - slight preference for open strings
        def note_score(sf):
            s, f = sf
            return (
                f,                 # low fret best
                0 if f == 0 else 1, # open slightly preferred
                abs(s - 3.5),       # middle strings slightly preferred
            )

        poss_sorted = sorted(poss, key=note_score)
        per.append(poss_sorted[:per_pitch_k])
    return per

def generate_voicings_for_event(
    pitches: List[int],
    max_fret: int = 20,
    per_pitch_k: int = 4,
    chord_k: int = 40
) -> List[Voicing]:
    """
    Generate top chord_k voicings for this event.
    Constraint: no two notes on the same string.
    """
    # sort pitches to keep deterministic mapping
    pitches_sorted = sorted([int(p) for p in pitches])
    per = per_pitch_candidates(pitches_sorted, max_fret=max_fret, per_pitch_k=per_pitch_k)
    if any(len(cands) == 0 for cands in per):
        return []

    # Backtracking with pruning
    best: List[Voicing] = []

    used_strings = set()
    chosen_strings: List[int] = []
    chosen_frets: List[int] = []

    def local_cost_for_assignment(strings: List[int], frets: List[int]) -> float:
        """
        Local cost = chord ergonomics only (no temporal transition).
        """
        span = chord_span_ignore_open(frets)
        anc = anchor_pos(frets)
        avgp = avg_fretted_pos(frets)

        # Penalties
        w_span = 1.2
        w_high = 0.25
        w_open_bonus = -0.05

        high_pen = sum(max(0, f - 12) for f in frets)  # discourage very high positions
        open_bonus = sum(1 for f in frets if f == 0)

        # chord size doesn't exceed 6 by construction (strings unique)
        cost = (w_span * span) + (w_high * high_pen) + (w_open_bonus * open_bonus)

        # mild penalty for "scattered strings" (jumping between far-apart strings in a chord)
        # (not super physical but helps)
        if strings:
            cost += 0.10 * (max(strings) - min(strings))
        return float(cost)

    def push_best(v: Voicing):
        best.append(v)
        best.sort(key=lambda x: x.local_cost)
        if len(best) > chord_k:
            best.pop()

    # A simple optimistic bound to prune: span can't be negative; we can prune if current cost already worse than worst.
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

        # quick prune
        if len(best) >= chord_k:
            # optimistic bound: current partial cost ~0; compare against current worst
            worst = best[-1].local_cost
            # compute a partial heuristic lower bound from already chosen frets
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

    # Final sort by local cost
    return sorted(best, key=lambda x: x.local_cost)

# ----------------------------
# Viterbi (beam search)
# ----------------------------
@dataclass
class BeamNode:
    score: float
    voicing: Voicing
    prev_index: Optional[int]  # index in previous beam list

def transition_cost(prev: Voicing, cur: Voicing) -> float:
    """
    Temporal ergonomics: how hard is it to move from prev -> cur
    """
    # Anchor (min fretted) is a decent proxy for hand position
    jump = abs(cur.anchor - prev.anchor)

    # Additional mild terms
    span_change = abs(cur.span - prev.span)
    avg_jump = abs(cur.avgpos - prev.avgpos)

    # Encourage staying in similar region; discourage huge leaps
    w_jump = 1.5
    w_avg = 0.5
    w_span_change = 0.2

    # slight penalty if chord uses very different string range (helps continuity)
    prev_str_range = max(prev.strings) - min(prev.strings) if prev.strings else 0
    cur_str_range = max(cur.strings) - min(cur.strings) if cur.strings else 0
    str_range_change = abs(cur_str_range - prev_str_range)
    w_str_range_change = 0.15

    return float(w_jump * jump + w_avg * avg_jump + w_span_change * span_change + w_str_range_change * str_range_change)

def viterbi_decode(
    events: List[Dict[str, Any]],
    max_fret: int = 20,
    per_pitch_k: int = 4,
    chord_k: int = 40,
    beam_size: int = 30
) -> List[Voicing]:
    """
    Returns best voicing sequence (one per event).
    """
    # Precompute candidates per event
    candidates: List[List[Voicing]] = []
    for ev in events:
        pitches = ev.get("pitches") or [n["pitch"] for n in ev.get("notes", [])]
        pitches = [int(p) for p in pitches]
        voicings = generate_voicings_for_event(pitches, max_fret=max_fret, per_pitch_k=per_pitch_k, chord_k=chord_k)
        if not voicings:
            # No candidates: fallback to empty (shouldn't happen if pitches are within guitar range)
            voicings = []
        candidates.append(voicings)

    # Initialize beam for t=0
    beam: List[BeamNode] = []
    for v in candidates[0]:
        beam.append(BeamNode(score=v.local_cost, voicing=v, prev_index=None))
    beam.sort(key=lambda n: n.score)
    beam = beam[:beam_size]

    backpointers: List[List[Optional[int]]] = [[n.prev_index for n in beam]]
    beams: List[List[BeamNode]] = [beam]

    # Iterate
    for t in range(1, len(events)):
        new_beam: List[BeamNode] = []
        cur_cands = candidates[t]

        # If empty, keep previous (shouldn't occur; but handle gracefully)
        if not cur_cands:
            # carry over with no change
            new_beam = [BeamNode(score=n.score, voicing=n.voicing, prev_index=i) for i, n in enumerate(beam)]
            new_beam.sort(key=lambda n: n.score)
            new_beam = new_beam[:beam_size]
            beam = new_beam
            beams.append(beam)
            backpointers.append([n.prev_index for n in beam])
            continue

        for j, cur_v in enumerate(cur_cands):
            # Find best predecessor among current beam
            best_score = float("inf")
            best_prev = None
            for i, prev_node in enumerate(beam):
                sc = prev_node.score + transition_cost(prev_node.voicing, cur_v) + cur_v.local_cost
                if sc < best_score:
                    best_score = sc
                    best_prev = i
            new_beam.append(BeamNode(score=best_score, voicing=cur_v, prev_index=best_prev))

        new_beam.sort(key=lambda n: n.score)
        new_beam = new_beam[:beam_size]

        beam = new_beam
        beams.append(beam)
        backpointers.append([n.prev_index for n in beam])

    # Reconstruct best path
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

    return {
        "note_acc": float(note_acc),
        "chord_exact_acc": float(chord_acc),
        "pred_span_mean": float(sum(pred_spans) / len(pred_spans)) if pred_spans else 0.0,
        "pred_span_p95": float(sorted(pred_spans)[int(0.95*(len(pred_spans)-1))]) if len(pred_spans) > 1 else (pred_spans[0] if pred_spans else 0.0),
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
    # ensure sorted
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
def run_folder(
    jsonl_dir: str,
    out_pred_dir: str = "./viterbi_preds",
    max_fret: int = 20,
    per_pitch_k: int = 4,
    chord_k: int = 40,
    beam_size: int = 30,
    limit_files: Optional[int] = None
) -> None:
    files = sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl")))
    if limit_files is not None:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"No jsonl files found in {jsonl_dir}")

    metrics_all = []
    for fp in files:
        events = load_events_from_jsonl(fp)
        preds = viterbi_decode(
            events,
            max_fret=max_fret,
            per_pitch_k=per_pitch_k,
            chord_k=chord_k,
            beam_size=beam_size
        )
        m = evaluate_piece(events, preds)
        m["file"] = os.path.basename(fp)
        metrics_all.append(m)

        out_path = os.path.join(out_pred_dir, os.path.basename(fp))
        save_predicted_jsonl(events, preds, out_path)

        print(f"[OK] {os.path.basename(fp)}  note_acc={m['note_acc']:.3f}  chord_exact={m['chord_exact_acc']:.3f}  jump_max={m['pred_jump_max']:.1f}  span_p95={m['pred_span_p95']:.1f}")

    # Aggregate
    note_accs = [x["note_acc"] for x in metrics_all]
    chord_accs = [x["chord_exact_acc"] for x in metrics_all]
    print("\n=== Aggregate ===")
    print(f"Files: {len(metrics_all)}")
    print(f"Mean note_acc:      {sum(note_accs)/len(note_accs):.3f}")
    print(f"Mean chord_exact:   {sum(chord_accs)/len(chord_accs):.3f}")

    # Worst 10 by note_acc
    metrics_all.sort(key=lambda x: x["note_acc"])
    print("\n=== Worst 10 (note_acc) ===")
    for x in metrics_all[:10]:
        print(f"{x['file']:40s} note_acc={x['note_acc']:.3f} chord_exact={x['chord_exact_acc']:.3f} jump_max={x['pred_jump_max']:.1f} span_p95={x['pred_span_p95']:.1f}")

if __name__ == "__main__":
    # Example:
    # python step3_viterbi_baseline.py
    JSONL_DIR = "./test_input_folder"
    OUT_PRED_DIR = "./test_output_folder"

    run_folder(
        jsonl_dir=JSONL_DIR,
        out_pred_dir=OUT_PRED_DIR,
        max_fret=20,
        per_pitch_k=4,
        chord_k=50,
        beam_size=30,
        limit_files=50,  # set e.g. 50 to test fast
    )
