import guitarpro
import os
import pretty_midi
import json

def parse_gp3_file(file_path):
    song = guitarpro.parse(file_path)
    tab_data = []
    for track in song.tracks:
        if not track.isPercussionTrack:
            for measure in track.measures:
                for voice in measure.voices:
                    for beat in voice.beats:
                        for note in beat.notes:
                            if note.string > 0:  # Ensure valid string
                                tab_data.append({
                                    'string': note.string,
                                    'fret': note.value,
                                    'time': beat.start / song.tempo  # Normalized time
                                })
    return tab_data

def parse_midi_file(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    note_data = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note_data.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'velocity': note.velocity
            })
    return note_data

def map_midi_to_tab(midi_data, tab_data):
    aligned_data = []
    for midi_note in midi_data:
        closest_tab_note = min(
            tab_data,
            key=lambda x: abs(x['time'] - midi_note['start'])  # Match by timing
        )
        aligned_data.append({
            'midi_pitch': midi_note['pitch'],
            'midi_time': midi_note['start'],
            'tab_string': closest_tab_note['string'],
            'tab_fret': closest_tab_note['fret']
        })
    return aligned_data


gp3_folder = 'data_E_standard'
midi_folder = 'labels'

tab_midi_pairs = []

for gp3_file in os.listdir(gp3_folder):
    if gp3_file.endswith('.gp3'):
        midi_file = os.path.join(midi_folder, gp3_file + '.mid')
        if os.path.exists(midi_file):
            tab_data = parse_gp3_file(os.path.join(gp3_folder, gp3_file))
            midi_data = parse_midi_file(midi_file)
            tab_midi_pairs.append({'gp3': tab_data, 'midi': midi_data})

output_file = 'aligned_data.json'
aligned_dataset = []

for pair in tab_midi_pairs:
    aligned_data = map_midi_to_tab(pair['midi'], pair['gp3'])
    aligned_dataset.extend(aligned_data)

with open(output_file, 'w') as f:
    json.dump(aligned_dataset, f)