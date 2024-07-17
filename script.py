import guitarpro
from midiutil import MIDIFile

# Function to convert a single track to a MIDI file
def track_to_midi(track: guitarpro.Track, filename: str, tempo: int):
    midi = MIDIFile(1)  # Create a single-track MIDI file
    midi.addTempo(0, 0, tempo)  # Set the tempo to 120 BPM

    # Add notes to the MIDI file
    time = 0
    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                for note in beat.notes:
                    midi.addNote(
                        0,  # Track number
                        0,  # Channel
                        note.realValue,  # Note pitch
                        time,  # Start time
                        beat.duration.time/960,  # Duration
                        note.velocity  # Volume
                    )
                time += beat.duration.time/960

    # Write the MIDI file to disk
    with open(filename, "wb") as output_file:
        midi.writeFile(output_file)

# Load the Guitar Pro file
gp_file = guitarpro.parse('./data/10_track_2_Chuck Berry (Riffs and Solos).gp3')

# Print track names and note details
# Convert each track to a MIDI file

non_guitar_instruments = [
    'drums', 'drumkit', 'vocals', 'piano', 'keyboard', 'synthesizer', 
    'bass', 'violin', 'viola', 'cello', 'double bass', 'flute', 'clarinet', 
    'oboe', 'bassoon', 'saxophone', 'trumpet', 'trombone', 'french horn', 
    'tuba', 'harp', 'xylophone', 'marimba', 'vibraphone', 'timpani', 
    'percussion', 'organ', 'accordion', 'harmonica'
]

for track in gp_file.tracks:
    print(track.strings)
    if not track.isPercussionTrack and not track.isBanjoTrack and not track.is12StringedGuitarTrack and not any(substring in track.name.lower() for substring in non_guitar_instruments):
        track_filename = f"./track_{track.number}_{track.name}.mid"
        track_to_midi(track, track_filename, gp_file.tempo)
        print(f"Track {track.number} ({track.name}) written to {track_filename}")
