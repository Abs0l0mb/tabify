import os
import json
import glob
import hashlib
from fractions import Fraction
from typing import List, Dict, Any

import guitarpro

E_STD_OPEN_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

def note_to_pitch_e_std(string: int, fret: int) -> int:
    return E_STD_OPEN_MIDI[string] + fret

def to_jsonable(x):
    if isinstance(x, Fraction):
        return float(x)  # ou int(x) si tu es sûr
    return x

def json_default(obj):
    from fractions import Fraction
    if isinstance(obj, Fraction):
        return float(obj)
    raise TypeError(f"{type(obj)} not serializable")

def dedup_notes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uniq = {}
    for n in notes:
        key = (n["string"], n["fret"])
        if key not in uniq or n["dur"] > uniq[key]["dur"]:
            uniq[key] = n
    return sorted(uniq.values(), key=lambda x: (x["pitch"], x["string"]))

def parse_gp3_voice0_events(gp_path: str) -> List[Dict[str, Any]]:
    song = guitarpro.parse(gp_path)
    if not song.tracks:
        return []

    track = song.tracks[0]
    events: List[Dict[str, Any]] = []
    current_time = 0

    for measure in track.measures:
        if not measure.voices:
            continue
        voice = measure.voices[0]

        for beat in voice.beats:
            start = current_time
            beat_dur = to_jsonable(beat.duration.time)

            notes = []
            for note in beat.notes:
                string = int(note.string)
                fret = int(note.value)
                pitch = note_to_pitch_e_std(string, fret)

                dur_percent = float(getattr(note, "durationPercent", 1.0))
                note_dur = int(float(beat_dur) * dur_percent)

                notes.append({
                    "string": string,
                    "fret": fret,
                    "pitch": pitch,
                    "dur": note_dur,
                    "dur_percent": dur_percent,
                })

            if notes:
                notes = dedup_notes(notes)
                events.append({"start": start, "dur": beat_dur, "notes": notes})

            current_time += beat_dur

    # tri explicite (safe)
    events.sort(key=lambda e: e["start"])
    return events

def stable_piece_id(gp_path: str) -> str:
    """
    ID stable court (utile pour train/test split, logs).
    """
    base = os.path.basename(gp_path)
    h = hashlib.sha1(gp_path.encode("utf-8")).hexdigest()[:10]
    return f"{os.path.splitext(base)[0]}__{h}"

def export_piece_to_jsonl(gp_path: str, out_jsonl_path: str) -> int:
    events = parse_gp3_voice0_events(gp_path)
    piece_id = stable_piece_id(gp_path)

    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)

    n_written = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for idx, ev in enumerate(events):
            # champs "plats" pour faciliter l’entraînement
            pitches = [n["pitch"] for n in ev["notes"]]
            strings = [n["string"] for n in ev["notes"]]
            frets = [n["fret"] for n in ev["notes"]]

            row = {
                "piece_id": piece_id,
                "event_idx": idx,
                "start": ev["start"],
                "dur": ev["dur"],
                "pitches": pitches,
                "strings": strings,
                "frets": frets,
                # on garde aussi la structure complète si tu veux debug
                "notes": ev["notes"],
            }
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
            n_written += 1

    return n_written

def export_dataset(gp_glob: str, out_dir: str) -> None:
    gp_files = sorted(glob.glob(gp_glob))
    if not gp_files:
        raise FileNotFoundError(f"Aucun fichier trouvé avec le pattern: {gp_glob}")

    os.makedirs(out_dir, exist_ok=True)

    total_events = 0
    for gp_path in gp_files:
        piece_name = os.path.splitext(os.path.basename(gp_path))[0]
        out_path = os.path.join(out_dir, f"{piece_name}.jsonl")
        n = export_piece_to_jsonl(gp_path, out_path)
        total_events += n
        print(f"[OK] {os.path.basename(gp_path)} -> {os.path.basename(out_path)} ({n} events)")

    print(f"\nTerminé. {len(gp_files)} morceaux, {total_events} events.")

if __name__ == "__main__":
    # Exemple :
    # - si tes gp3 sont dans data/gp3/*.gp3
    # - sortie dans data/jsonl/
    export_dataset(gp_glob="./3-only_E_standard/*.gp3", out_dir="./jsonl_dataset")
