from math import gcd
from functools import reduce
import mido

def midi_min_step_in_gp(mid_path: str, gp_quarter_ticks: int = 960):
    mid = mido.MidiFile(mid_path)
    tpq = mid.ticks_per_beat

    merged = mido.merge_tracks(mid.tracks)
    abs_tick = 0
    times = []
    for msg in merged:
        abs_tick += msg.time
        if msg.type in ("note_on", "note_off"):
            times.append(abs_tick)

    times = sorted(set(times))
    deltas = [b - a for a, b in zip(times, times[1:]) if b - a > 0]
    if not deltas:
        return None

    g = reduce(gcd, deltas)  # minimal grid in MIDI ticks
    # convert this to GP ticks (can be fractional)
    gp_step = g * gp_quarter_ticks / tpq
    return tpq, g, gp_step

# Exemple:
print(midi_min_step_in_gp("./midi_originals/hearts_clockwork.mid"))