"""
gp_to_midi.py — Convert a Guitar Pro file (.gp3/.gp4/.gp5/.gpx) to MIDI.

Usage:
    python gp_to_midi.py --input song.gp5 --output song.mid
    python gp_to_midi.py --input song.gp5 --output song.mid --track 0
    python gp_to_midi.py --input song.gp5 --list-tracks

Can also be imported:
    from gp_to_midi import gp_to_midi
    gp_to_midi("song.gp5", "song.mid")
"""

import argparse
import os
from typing import List, Optional

import guitarpro
from guitarpro import models
from midiutil import MIDIFile


# Instrument names that are not guitar — used for auto track selection
_NON_GUITAR_KEYWORDS = {
    "drum", "bass", "vocal", "voice", "piano", "keyboard", "synth",
    "violin", "viola", "cello", "flute", "clarinet", "oboe", "bassoon",
    "saxophone", "trumpet", "trombone", "horn", "tuba", "harp",
    "organ", "accordion", "harmonica", "xylophone", "marimba",
}


def _is_guitar_track(track: models.Track) -> bool:
    if track.isPercussionTrack or track.isBanjoTrack:
        return False
    name = track.name.lower()
    return not any(kw in name for kw in _NON_GUITAR_KEYWORDS)


def list_tracks(gp_path: str) -> None:
    """Print all tracks in a GP file with their index and name."""
    song = guitarpro.parse(gp_path)
    print(f"Tracks in {os.path.basename(gp_path)}:")
    for i, track in enumerate(song.tracks):
        flags = []
        if track.isPercussionTrack:
            flags.append("percussion")
        if track.isBanjoTrack:
            flags.append("banjo")
        if _is_guitar_track(track):
            flags.append("guitar [auto]")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  [{i}] {track.name}{flag_str}")


def _measure_start_times(song: models.Song) -> List[float]:
    """Return the start time (in quarter-note beats) for each measure."""
    QUARTER_TICKS = 960
    starts = []
    t = 0.0
    for header in song.measureHeaders:
        starts.append(t)
        ts = header.timeSignature
        measure_ticks = ts.numerator * QUARTER_TICKS * 4.0 / ts.denominator.value
        t += measure_ticks / QUARTER_TICKS  # convert to beats
    return starts


def gp_to_midi(
    gp_path: str,
    out_midi_path: str,
    track_index: Optional[int] = None,
) -> str:
    """
    Convert a single track from a GP file to a MIDI file.

    Args:
        gp_path:       Path to the input GP file.
        out_midi_path: Path for the output .mid file.
        track_index:   Which track to export (0-based). If None, auto-selects
                       the first guitar track.

    Returns:
        out_midi_path
    """
    song = guitarpro.parse(gp_path)

    # Track selection
    if track_index is not None:
        if track_index >= len(song.tracks):
            raise ValueError(f"Track index {track_index} out of range (file has {len(song.tracks)} tracks)")
        track = song.tracks[track_index]
    else:
        guitar_tracks = [t for t in song.tracks if _is_guitar_track(t)]
        if not guitar_tracks:
            raise ValueError("No guitar track found. Use --list-tracks and specify --track manually.")
        track = guitar_tracks[0]

    measure_starts = _measure_start_times(song)

    midi = MIDIFile(1)
    midi.addTempo(0, 0, song.tempo)

    for mi, measure in enumerate(track.measures):
        if mi >= len(measure_starts):
            break
        m_start = measure_starts[mi]

        for voice in measure.voices:
            voice_time = m_start  # each voice starts at measure begin
            for beat in voice.beats:
                beat_dur = beat.duration.time / 960.0  # ticks → quarter beats
                if beat.status != models.BeatStatus.rest:
                    for note in beat.notes:
                        # Skip tied notes — they're continuations of a previous note
                        if hasattr(models, "NoteType") and note.type == models.NoteType.tie:
                            continue
                        if note.type == 2:  # GP5 tie value fallback
                            continue
                        velocity = max(1, min(127, note.velocity))
                        midi.addNote(
                            track=0,
                            channel=0,
                            pitch=note.realValue,
                            time=voice_time,
                            duration=beat_dur,
                            volume=velocity,
                        )
                voice_time += beat_dur

    os.makedirs(os.path.dirname(out_midi_path) or ".", exist_ok=True)
    with open(out_midi_path, "wb") as f:
        midi.writeFile(f)

    return out_midi_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert a Guitar Pro file to MIDI")
    ap.add_argument("--input",       required=True, help="Input GP file (.gp3/.gp4/.gp5/.gpx)")
    ap.add_argument("--output",      help="Output .mid file (default: same name as input)")
    ap.add_argument("--track",       type=int, default=None, help="Track index to export (0-based, default: auto)")
    ap.add_argument("--list-tracks", action="store_true", help="List all tracks and exit")
    args = ap.parse_args()

    if args.list_tracks:
        list_tracks(args.input)
    else:
        out = args.output or os.path.splitext(args.input)[0] + ".mid"
        result = gp_to_midi(args.input, out, track_index=args.track)
        print(f"[OK] wrote {result}")
