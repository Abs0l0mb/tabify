import guitarpro
import os
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
                    velocity = max(0, min(127, note.velocity))  # Clamp velocity between 0 and 127
                    midi.addNote(
                        0,  # Track number
                        0,  # Channel
                        note.realValue,  # Note pitch
                        time,  # Start time
                        beat.duration.time/960,  # Duration
                        velocity  # Volume
                    )
                time += beat.duration.time/960

    # Write the MIDI file to disk
    with open(filename, "wb") as output_file:
        midi.writeFile(output_file)


non_guitar_instruments = [
    'drums', 'drumkit', 'vocals', 'piano', 'keyboard', 'synthesizer', 
    'bass', 'violin', 'viola', 'cello', 'double bass', 'flute', 'clarinet', 
    'oboe', 'bassoon', 'saxophone', 'trumpet', 'trombone', 'french horn', 
    'tuba', 'harp', 'xylophone', 'marimba', 'vibraphone', 'timpani', 
    'percussion', 'organ', 'accordion', 'harmonica'
]

gp3_file_list = [f for f in os.listdir("./data_E_standard") if f.endswith('.gp3')]

for file in gp3_file_list:

    gp_file = guitarpro.parse(f"./data_E_standard/{file}")

    for track in gp_file.tracks:
        if not track.isPercussionTrack and not track.isBanjoTrack and not track.is12StringedGuitarTrack and not any(substring in track.name.lower() for substring in non_guitar_instruments):
            track_filename = f"./labels/{file}.mid" # We already pre processed the dataset so that there is only one track per gp3 file, so this will be unique
            track_to_midi(track, track_filename, gp_file.tempo)
            print(f"Track {file} written to {track_filename}")

