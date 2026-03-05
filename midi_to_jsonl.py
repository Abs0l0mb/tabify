import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from fractions import Fraction

import mido


def stable_piece_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def quantize_nearest(t: Fraction, step: int) -> int:
    stepF = Fraction(step, 1)
    q = t / stepF
    n, d = q.numerator, q.denominator
    k, r = divmod(n, d)
    if 2 * r >= d:
        k += 1
    return int(k * step)

def quantize_floor(t: Fraction, step: int) -> int:
    stepF = Fraction(step, 1)
    q = t / stepF
    n, d = q.numerator, q.denominator
    k = n // d
    return int(k * step)


def midi_to_jsonl_active_compressed(
    midi_path: str,
    out_path: str,
    gp_quarter_ticks: int = 960,
    step: int = 120,               # 120 => 1/32 note if quarter=960
    min_pitch: int = 40,
    max_pitch: int = 88,
    drop_drum_channel_10: bool = True,
):
    # Safety: step should divide quarter for "clean" notation
    if gp_quarter_ticks % step != 0:
        print(f"[WARN] step={step} does not divide quarter={gp_quarter_ticks}. Consider 120, 160, 240, ...")

    mid = mido.MidiFile(midi_path)
    tpq = mid.ticks_per_beat

    # Exact scale factor: GP ticks per MIDI tick
    scale = Fraction(gp_quarter_ticks, tpq)

    merged = mido.merge_tracks(mid.tracks)

    # Collect state changes at quantized grid times: at[t] = list of ('on'/'off', pitch)
    abs_tick = 0
    at: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

    for msg in merged:
        abs_tick += msg.time
        if msg.type not in ("note_on", "note_off"):
            continue
        if drop_drum_channel_10 and hasattr(msg, "channel") and msg.channel == 9:
            continue

        pitch = int(msg.note)
        if not (min_pitch <= pitch <= max_pitch):
            continue

        t_gp = Fraction(abs_tick, 1) * scale

        if msg.type == "note_on" and msg.velocity > 0:
            t_q = quantize_nearest(t_gp, step)
            at[t_q].append(("on", pitch))
        else:
            t_q = quantize_floor(t_gp, step)   # ✅ crucial
            at[t_q].append(("off", pitch))  

    if not at:
        raise SystemExit("No note events found (after filtering).")

    # Determine grid range
    t_min = min(at.keys())
    t_max = max(at.keys())

    active: Set[int] = set()
    grid_states: List[Tuple[int, Tuple[int, ...]]] = []

    t = t_min
    while t <= t_max:
        for typ, p in at.get(t, []):
            if typ == "on":
                active.add(p)
            else:
                active.discard(p)
        grid_states.append((t, tuple(sorted(active))))
        t += step

    # Compress consecutive identical states
    segments = []
    seg_start, seg_state = grid_states[0]
    seg_len = step
    for (t, st) in grid_states[1:]:
        if st == seg_state:
            seg_len += step
        else:
            segments.append((seg_start, seg_len, seg_state))
            seg_start, seg_state = t, st
            seg_len = step
    segments.append((seg_start, seg_len, seg_state))

    # Write JSONL
    piece_id = stable_piece_id(midi_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (start, dur, pitches) in enumerate(segments):
            row = {
                "piece_id": piece_id,
                "event_idx": idx,
                "start": float(start),
                "dur": float(dur),
                "pitches": list(pitches),  # active pitches in this segment (empty => rest)
                "notes": []
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] {midi_path} -> {out_path}  segments={len(segments)}  tpq={tpq}  step={step}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=120)
    ap.add_argument("--gpq", type=int, default=960)
    args = ap.parse_args()
    
    midi_to_jsonl_active_compressed(
        midi_path=args.midi,
        out_path=args.out,
        gp_quarter_ticks=args.gpq,
        step=args.step,
    )
