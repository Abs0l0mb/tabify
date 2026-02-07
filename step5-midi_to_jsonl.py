import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import mido


# --- helpers ---
def stable_piece_id(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def quantize(x: float, q: int) -> int:
    """Quantize x to nearest multiple of q."""
    return int(round(x / q) * q)


def clamp_pitch_range(p: int, lo: int, hi: int) -> bool:
    return lo <= p <= hi


def parse_midi_onsets(
    midi_path: str,
    gp_quarter_ticks: int = 960,
    quantize_to: int = 240,
    min_pitch: int = 40,
    max_pitch: int = 88,
    drop_drum_channel_10: bool = True,
) -> Tuple[List[Dict], int]:
    """
    Convert MIDI into a list of onset events:
      - group notes starting at the same absolute tick
      - compute event 'dur' = time to next onset
      - convert MIDI ticks -> GP ticks (quarter=960) using scale = 960 / ticks_per_beat
      - quantize starts and durs to quantize_to (e.g. 240 = 1/16)

    Returns: (events, ticks_per_beat)
    """
    mid = mido.MidiFile(midi_path)
    tpq = mid.ticks_per_beat  # MIDI PPQ
    scale = gp_quarter_ticks / float(tpq)

    # Merge all tracks into one timeline of messages
    merged = mido.merge_tracks(mid.tracks)

    abs_tick = 0
    onsets_by_tick: Dict[int, List[int]] = defaultdict(list)

    for msg in merged:
        abs_tick += msg.time  # msg.time is delta ticks in merged track
        if msg.type == "note_on" and msg.velocity > 0:
            if drop_drum_channel_10 and hasattr(msg, "channel") and msg.channel == 9:
                continue
            pitch = int(msg.note)
            if clamp_pitch_range(pitch, min_pitch, max_pitch):
                onsets_by_tick[abs_tick].append(pitch)

    if not onsets_by_tick:
        return [], tpq

    # Sort unique onset times
    onset_ticks = sorted(onsets_by_tick.keys())

    # Build events in GP tick domain
    events = []
    for i, t0 in enumerate(onset_ticks):
        t1 = onset_ticks[i + 1] if i + 1 < len(onset_ticks) else None

        start_gp = t0 * scale
        if t1 is None:
            # last event: set a default duration (1 quarter) if nothing else
            dur_gp = gp_quarter_ticks
        else:
            dur_gp = (t1 - t0) * scale

        start_q = quantize(start_gp, quantize_to)
        dur_q = max(quantize(dur_gp, quantize_to), quantize_to)  # avoid 0

        pitches = sorted(set(onsets_by_tick[t0]))  # chord pitches, unique + sorted

        events.append(
            {
                "start": float(start_q),
                "dur": float(dur_q),
                "pitches": pitches,
            }
        )

    # Fix potential duplicates introduced by quantization (same start after quantize)
    # Merge them by unioning pitches and taking max duration.
    merged_events = []
    for ev in events:
        if merged_events and merged_events[-1]["start"] == ev["start"]:
            merged_events[-1]["pitches"] = sorted(set(merged_events[-1]["pitches"]) | set(ev["pitches"]))
            merged_events[-1]["dur"] = max(float(merged_events[-1]["dur"]), float(ev["dur"]))
        else:
            merged_events.append(ev)

    # Recompute durations from successive starts after quantization to keep consistency
    # (dur = next_start - start)
    for i in range(len(merged_events) - 1):
        d = float(merged_events[i + 1]["start"]) - float(merged_events[i]["start"])
        merged_events[i]["dur"] = max(d, float(quantize_to))
    # last dur unchanged (keeps default)

    # Add event_idx + piece_id later when writing
    return merged_events, tpq


def write_jsonl(events: List[Dict], out_path: str, piece_id: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, ev in enumerate(events):
            row = {
                "piece_id": piece_id,
                "event_idx": idx,
                "start": ev["start"],
                "dur": ev["dur"],
                "pitches": ev["pitches"],
                # on laisse "notes" vide: Viterbi lit d'abord "pitches"
                "notes": [],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True, help="Path to input .mid")
    ap.add_argument("--out", required=True, help="Path to output .jsonl")
    ap.add_argument("--quant", type=int, default=240, help="Quantization in GP ticks (default=240 = 1/16)")
    ap.add_argument("--gpq", type=int, default=960, help="GP quarter ticks (default=960)")
    ap.add_argument("--min_pitch", type=int, default=40, help="Min pitch kept (default=40 E2)")
    ap.add_argument("--max_pitch", type=int, default=88, help="Max pitch kept (default=88 E6-ish)")
    args = ap.parse_args()

    piece_id = stable_piece_id(args.midi)
    events, tpq = parse_midi_onsets(
        args.midi,
        gp_quarter_ticks=args.gpq,
        quantize_to=args.quant,
        min_pitch=args.min_pitch,
        max_pitch=args.max_pitch,
    )

    if not events:
        raise SystemExit("No onset events found (after filtering).")

    write_jsonl(events, args.out, piece_id=piece_id)
    print(f"[OK] {args.midi} -> {args.out}")
    print(f"     MIDI ticks_per_beat={tpq}, GP quarter_ticks={args.gpq}, quant={args.quant}, events={len(events)}")


if __name__ == "__main__":
    main()
