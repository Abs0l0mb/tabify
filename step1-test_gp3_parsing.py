import guitarpro
from typing import List, Dict, Any

# E standard, MIDI numbers des cordes à vide (1 = mi aigu, 6 = mi grave)
E_STD_OPEN_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

def note_to_pitch_e_std(string: int, fret: int) -> int:
    return E_STD_OPEN_MIDI[string] + fret

def dedup_notes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Supprime les doublons au sein d’un même beat.
    On garde l’occurrence avec la plus grande durée si jamais il y a conflit.
    """
    uniq = {}
    for n in notes:
        key = (n["string"], n["fret"])  # robuste: même position => même note
        if key not in uniq or n["dur"] > uniq[key]["dur"]:
            uniq[key] = n
    # tri: du grave vers l’aigu (pitch croissant)
    return sorted(uniq.values(), key=lambda x: (x["pitch"], x["string"]))

def parse_gp3_voice0_events(gp_path: str) -> List[Dict[str, Any]]:
    song = guitarpro.parse(gp_path)

    if not song.tracks:
        return []

    track = song.tracks[0]
    events: List[Dict[str, Any]] = []

    current_time = 0  # temps absolu "piste" en unités GP

    for measure in track.measures:
        # Voice 0 only
        if not measure.voices:
            continue
        voice = measure.voices[0]

        for beat in voice.beats:
            start = current_time
            beat_dur = beat.duration.time

            notes = []
            for note in beat.notes:
                string = int(note.string)
                fret = int(note.value)
                pitch = note_to_pitch_e_std(string, fret)

                dur_percent = float(getattr(note, "durationPercent", 1.0))
                note_dur = int(beat_dur * dur_percent)

                notes.append(
                    {
                        "string": string,
                        "fret": fret,
                        "pitch": pitch,
                        "dur": note_dur,
                        "dur_percent": dur_percent,
                    }
                )

            # même si notes vide, on peut garder l’event (silence) si tu veux.
            # Pour l’instant on garde seulement s’il y a au moins une note:
            if notes:
                notes = dedup_notes(notes)
                events.append({"start": start, "dur": beat_dur, "notes": notes})

            current_time += beat_dur

    return events

if __name__ == "__main__":
    gp_path = "./2_track_3_Noel Gallagher - Electric Guitar.gp3"
    events = parse_gp3_voice0_events(gp_path)
    for e in events[:15]:
        print(e)



