import os
import json
import math
from typing import List, Dict, Any, Tuple

import guitarpro
from guitarpro import models


def load_pred_events(pred_jsonl_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(pred_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if rows and "event_idx" in rows[0]:
        rows.sort(key=lambda r: int(r.get("event_idx", 0)))
    else:
        rows.sort(key=lambda r: float(r.get("start", 0.0)))
    return rows


def set_time_signature(header: models.MeasureHeader, numerator: int, denominator: int) -> None:
    header.timeSignature.numerator = int(numerator)
    header.timeSignature.denominator.value = int(denominator)
    # beams auto-init in TimeSignature.__attrs_post_init__ if empty


def measure_len_ticks(numerator: int, denominator: int, quarter_ticks: int = 960) -> int:
    # Duration.quarterTime = 960 in PyGuitarPro. :contentReference[oaicite:3]{index=3}
    return int(round(numerator * quarter_ticks * (4.0 / denominator)))


def ensure_single_guitar_track_e_standard(song: models.Song) -> models.Track:
    """
    Song() comes with 1 track by default. We reuse it and configure it.
    Track measures are tied to song.measureHeaders. :contentReference[oaicite:4]{index=4}
    """
    track = song.tracks[0]
    track.name = "Guitar"
    track.isPercussionTrack = False
    track.fretCount = 24
    track.channel.instrument = 29  # Overdriven Guitar (optional)

    # E standard: high E to low E
    track.strings = [
        models.GuitarString(number=1, value=64),  # E4
        models.GuitarString(number=2, value=59),  # B3
        models.GuitarString(number=3, value=55),  # G3
        models.GuitarString(number=4, value=50),  # D3
        models.GuitarString(number=5, value=45),  # A2
        models.GuitarString(number=6, value=40),  # E2
    ]
    return track


def add_note_to_beat(beat: models.Beat, string: int, fret: int, velocity: int = 90) -> None:
    n = models.Note(beat=beat)
    n.string = int(string)
    n.value = int(fret)
    n.velocity = int(velocity)
    beat.notes.append(n)


def build_gp5_from_pred_rows(
    pred_rows: List[Dict[str, Any]],
    out_gp5_path: str,
    title: str = "Viterbi prediction",
    tempo: int = 120,
    time_sig: Tuple[int, int] = (4, 4),
    quarter_ticks: int = 960,
) -> str:
    num, den = time_sig
    default_measure_len = measure_len_ticks(num, den, quarter_ticks=quarter_ticks)

    # IMPORTANT: use Song() default structure (it already has 1 header + 1 track)
    song = models.Song(versionTuple=(5, 1, 0))
    song.title = title
    song.tempo = int(tempo)

    track = ensure_single_guitar_track_e_standard(song)

    # Configure first measure header (already exists by default in Song.measureHeaders) :contentReference[oaicite:5]{index=5}
    first_header = song.measureHeaders[0]
    first_header.number = 1
    set_time_signature(first_header, num, den)

    # The track already has a Measure linked to that header (created by Track factory) :contentReference[oaicite:6]{index=6}
    current_measure_index = 0
    current_measure = track.measures[current_measure_index]

    # Make sure voices exist & are empty
    # Measure has 2 voices by default (maxVoices=2). :contentReference[oaicite:7]{index=7}
    for v in current_measure.voices:
        v.beats = []

    remaining = default_measure_len

    def open_new_measure(numerator=num, denominator=den) -> None:
        nonlocal current_measure_index, current_measure, remaining

        # This is the KEY: use song.newMeasure() so headers+measures stay consistent :contentReference[oaicite:8]{index=8}
        song.newMeasure()

        current_measure_index += 1
        current_measure = track.measures[current_measure_index]

        # header numbering is stored in the header object; ensure it’s correct
        header = current_measure.header
        header.number = current_measure_index + 1
        set_time_signature(header, numerator, denominator)

        for v in current_measure.voices:
            v.beats = []

        remaining = measure_len_ticks(numerator, denominator, quarter_ticks=quarter_ticks)

    for row in pred_rows:
        dur = row.get("dur")
        if dur is None:
            raise ValueError("Each row must contain 'dur'.")

        dur_ticks = int(round(float(dur)))
        if dur_ticks <= 0:
            continue

        pred_strings = row.get("pred_strings", [])
        pred_frets = row.get("pred_frets", [])
        if len(pred_strings) != len(pred_frets):
            raise ValueError(f"pred_strings/pred_frets mismatch at event_idx={row.get('event_idx')}")

        # Handle beats longer than a measure: allocate a dedicated bigger measure
        if dur_ticks > default_measure_len:
            # if current measure already has content, go next
            if remaining != default_measure_len:
                open_new_measure(num, den)

            unit = quarter_ticks * (4.0 / den)
            num_needed = int(math.ceil(dur_ticks / unit))
            open_new_measure(num_needed, den)

        # If it doesn't fit, open next measure
        if dur_ticks > remaining:
            open_new_measure(num, den)

        voice0 = current_measure.voices[0]
        beat = models.Beat(voice=voice0)
        beat.notes = []
        beat.duration = models.Duration.fromTime(dur_ticks)  # will raise if unrepresentable :contentReference[oaicite:9]{index=9}

        if len(pred_strings) == 0:   # ou pitches == [] si tu préfères
            beat.status = models.BeatStatus.rest
        else:
            beat.status = models.BeatStatus.normal

        # add notes (1 per string)
        used = set()
        for s, f in zip(pred_strings, pred_frets):
            s = int(s)
            f = int(f)
            if s in used:
                continue
            used.add(s)
            add_note_to_beat(beat, s, f)

        voice0.beats.append(beat)
        remaining -= dur_ticks

        # If measure perfectly filled, move on (helps layout)
        if remaining == 0:
            open_new_measure(num, den)

    # Remove trailing empty measure if created
    if track.measures and len(track.measures[-1].voices[0].beats) == 0:
        track.measures.pop()
        song.measureHeaders.pop()

    os.makedirs(os.path.dirname(out_gp5_path) or ".", exist_ok=True)
    guitarpro.write(song, out_gp5_path, version=(5, 1, 0))
    return out_gp5_path


def roundtrip_sanity_check(gp5_path: str, max_print_measures: int = 3) -> None:
    s = guitarpro.parse(gp5_path)
    t = s.tracks[0]
    print(f"[CHECK] measures={len(t.measures)} tempo={s.tempo}")
    for mi, m in enumerate(t.measures[:max_print_measures], start=1):
        v0 = m.voices[0]
        durs = [b.duration.time for b in v0.beats[:10]]
        print(f"  measure {mi}: beats={len(v0.beats)} first_durs={durs}")


if __name__ == "__main__":
    PRED_JSONL = "test_output_folder/hearts_clockwork.jsonl"
    OUT_GP5 = "./viterbi_preds_gp5/hearts_clockwork.gp5"

    rows = load_pred_events(PRED_JSONL)
    out = build_gp5_from_pred_rows(
        pred_rows=rows,
        out_gp5_path=OUT_GP5,
        title="Viterbi prediction",
        tempo=120,
        time_sig=(4, 4),
        quarter_ticks=960,
    )
    print(f"[OK] wrote {out}")
    roundtrip_sanity_check(out)
