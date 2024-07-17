import guitarpro

# Parse the original file
song = guitarpro.parse('./data/10_track_2_Chuck Berry (Riffs and Solos).gp3')

# Add notes to the MIDI file
for track in song.tracks:
    time = 0
    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                for note in beat.notes:
                    print(note.value, note.string, note.beat.duration.time, note.durationPercent)
