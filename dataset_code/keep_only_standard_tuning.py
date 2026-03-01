import os
import guitarpro as gp

# Define standard tuning notes in semitones from the lowest string (E2)
standard_tuning = [64, 59, 55, 50, 45, 40]

# Function to check if tuning is a relative standard tuning
def is_relative_standard_tuning(tuning):
    # Calculate the semitone difference from the standard tuning
    differences = [tuning[i] - standard_tuning[i] for i in range(6)]
    # Check if all differences are the same
    return all(diff == differences[0] for diff in differences), differences[0] if all(diff == differences[0] for diff in differences) else None

# Function to convert tuning to standard EADGBE tuning
def convert_to_standard(tuning, track):
    is_relative, semitone_shift = is_relative_standard_tuning(tuning)
    if is_relative:
        for string in track.strings:
            string.value -= semitone_shift
        return True
    return False

# Function to process each .gp3 file
def process_gp3_file(file_path, deleted, converted):
    song = gp.parse(file_path)
    track = song.tracks[0]  # Assuming only one track per file
    tuning = [string.value for string in track.strings]
    
    if(len(tuning) != 6):  
        os.remove(file_path)
        deleted += 1
        print(f"Deleted {file_path} (not in relative standard tuning).")
    elif is_relative_standard_tuning(tuning)[0]:
        if convert_to_standard(tuning, track):
            gp.write(song, file_path)
            converted += 1
            print(f"Converted {file_path} to standard tuning.")
    else:
        os.remove(file_path)
        deleted += 1
        print(f"Deleted {file_path} (not in relative standard tuning).")
        
    return (deleted, converted)

# Main script to iterate through all .gp3 files in a folder
def process_folder(folder_path, deleted, converted, total):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gp3"):
                file_path = os.path.join(root, file)
                print(total)
                total += 1
                (deleted, converted) = process_gp3_file(file_path, deleted, converted)

    return (deleted, converted, total)

deleted = 0
converted = 0
total = 0

(deleted, converted, total) = process_folder('./data_E_standard', deleted, converted, total)
print('Ended')
print('Converted files : ', converted)
print('Deleted files : ', deleted)
print('Total files processed : ', total)