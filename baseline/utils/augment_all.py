import os
import pandas as pd
import shutil
import random
from audiomentations import Compose, TimeMask, Gain, TimeStretch, Shift, AddGaussianNoise, PitchShift
import librosa
import soundfile as sf


def augment_audio(original_audio, sr):
    
    augment = Compose([
        # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
        # PitchShift(min_semitones=-4, max_semitones=4, p=1),
        # HighPassFilter(min_cutoff_freq=1500, max_cutoff_freq=2500, p=1),
        # TimeMask(min_band_part=0.1, max_band_part=0.5, p=1),
        Gain(min_gain_db=-12, max_gain_db=12, p=1),
        TimeStretch(min_rate=0.9, max_rate=1.2, p=1),
        Shift(min_shift=-0.5, max_shift=0.5, rollover=False, p=1),
        # ClippingDistortion(min_percentile_threshold=0,
        #    max_percentile_threshold=40, p=1),
        # LowPassFilter(min_cutoff_freq=1500, max_cutoff_freq=8000, p=1),
        # PolarityInversion(p=1)
    ])
    
    augmented_audio = augment(original_audio, sample_rate=sr)
    
    return augmented_audio

# Function to augment a file and save it
def augment_and_save(original_file, augmented_file):
    
    original_audio, sr = librosa.load(original_file, sr=22050, mono=True)
    
    augmented_audio = augment_audio(original_audio, sr=22050)

    sf.write(augmented_file, augmented_audio, samplerate=22050)
    
    

def process_files(audio_dir, metadata_file, augmented_dir, percentage):
    
    # Read the metadata CSV file
    metadata = pd.read_csv(metadata_file)

    # Create the augmented directory if it does not exist
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)

    if not os.path.exists(augmented_dir+"/audio"):
        os.makedirs(augmented_dir+"/audio")
    
    if not os.path.exists(augmented_dir+"metadata"):
        os.makedirs(augmented_dir+"/metadata")
    
    # New metadata list
    new_metadata = []

    # Process each class
    for class_id, group in metadata.groupby('Class ID'):
        
        files = group['Dataset File Name'].tolist()
        num_files_to_augment = int(len(files) * (percentage / 100.0))
        
        # Randomly choose files to augment
        files_to_augment = random.sample(files, num_files_to_augment)
        
        for file in files:
            
            # Copy the original file
            original_file_path = os.path.join(audio_dir, file)  # Modify with actual path
            new_original_file_path = os.path.join(augmented_dir, 'audio')
            shutil.copy(original_file_path, new_original_file_path)
            
            # Update metadata for original file
            class_name = group[group['Dataset File Name'] == file]['Class Name'].values[0]
            new_metadata.append([file, file, class_id, class_name])
            
            if file in files_to_augment:
                # Augment the file and save it
                augmented_file = file.replace('.wav', '_augmented.wav')
                augmented_file_path = os.path.join(augmented_dir, "audio", augmented_file)
                augment_and_save(original_file_path, augmented_file_path)
                
                # Update metadata for augmented file
                new_metadata.append([augmented_file, augmented_file, class_id, class_name])


    # Write new metadata CSV
    new_metadata_df = pd.DataFrame(new_metadata, columns=['Source File Name', 'Dataset File Name', 'Class ID', 'Class Name'])
    new_metadata_df.to_csv(os.path.join(augmented_dir, 'metadata', 'augmented_metadata_FSC22.csv'), index=False)

# Example usage
audio_dir = "../../data/raw/audio"
metadata_file = '../../data/raw/metadata/metadata_FSC22.csv'  # Replace with your actual path
augmented_dir = '../../data/augmented'
percentage = 100

process_files(audio_dir, metadata_file, augmented_dir, percentage)
