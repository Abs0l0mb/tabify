import os
import guitarpro
from concurrent.futures import ThreadPoolExecutor

def save_track_as_gp3(original_file_path, track, track_index, target_folder):
    # Parse the original file
    song = guitarpro.parse(original_file_path)
    
    # Remove all tracks except the one we want to keep
    song.tracks = [track]
    
    # Create a new filename for the track
    base_name = os.path.basename(original_file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    new_file_path = os.path.join(target_folder, f"{name_without_ext}_track_{track_index + 1}_{track.name}.gp3")
    print(new_file_path)
    
    # Save the new GP3 file with only this track
    guitarpro.write(song, new_file_path)

def process_single_file(file_path, target_folder, non_guitar_instruments):
    # Load the GP3 file
    song = guitarpro.parse(file_path)
    
    # Iterate over each track and save it as a separate GP3 file
    for index, track in enumerate(song.tracks):
        if not track.isPercussionTrack and not track.isBanjoTrack and not track.is12StringedGuitarTrack and not any(substring in track.name.lower() for substring in non_guitar_instruments):
            save_track_as_gp3(file_path, track, index, target_folder)

def process_gp3_files_in_folder(folder_path, target_folder):
    non_guitar_instruments = [
        'drums', 'drumkit', 'vocals', 'piano', 'keyboard', 'synthesizer', 
        'bass', 'violin', 'viola', 'cello', 'double bass', 'flute', 'clarinet', 
        'oboe', 'bassoon', 'saxophone', 'trumpet', 'trombone', 'french horn', 
        'tuba', 'harp', 'xylophone', 'marimba', 'vibraphone', 'timpani', 
        'percussion', 'organ', 'accordion', 'harmonica'
    ]

    # List all files in the folder
    files = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.gp3')]
    
    # Use ThreadPoolExecutor to parallelize processing
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, file_path, target_folder, non_guitar_instruments) for file_path in files]
        for future in futures:
            future.result()  # Wait for all futures to complete

# Specify the folder containing the GP3 files
source_folder_path = './tabs_converted'
target_folder_path = './data'

# Process all GP3 files in the specified folder
process_gp3_files_in_folder(source_folder_path, target_folder_path)