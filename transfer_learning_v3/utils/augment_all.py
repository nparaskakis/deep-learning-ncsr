import os
import pandas as pd
import shutil
import random
from audiomentations import Compose, TimeMask, Gain, TimeStretch, Shift, AddGaussianNoise, PitchShift
import librosa
import soundfile as sf


def augment_audio(original_audio, sr, augment_type):
    
    augment_type_A = Compose([
        Shift(min_shift=-0.5, max_shift=0.5, rollover=False, p=1),
    ])
    
    augment_type_B = Compose([
        Gain(min_gain_db=-12, max_gain_db=12, p=1),
        TimeStretch(min_rate=0.9, max_rate=1.2, p=1),
        Shift(min_shift=-0.5, max_shift=0.5, rollover=False, p=1),
    ])
    
    if augment_type == "type_A":
        augmented_audio = augment_type_A(original_audio, sample_rate=sr)
    elif augment_type == "type_B":
        augmented_audio = augment_type_B(original_audio, sample_rate=sr)
    else:
        raise ValueError("Error in type of augmentation.")
    
    return augmented_audio



def augment_and_save(original_file, augmented_file, augment_type):
    
    original_audio, sr = librosa.load(original_file, sr=22050, mono=True)
    
    if augment_type == "type_A":
        augmented_audio = augment_audio(original_audio, sr=22050, augment_type=augment_type)
    elif augment_type == "type_B":
        augmented_audio = augment_audio(original_audio, sr=22050, augment_type=augment_type)
    else:
        raise ValueError("Error in type of augmentation.")
    
    sf.write(augmented_file, augmented_audio, samplerate=22050)
        
    
    

def process_files(audio_dir, metadata_dir, augmented_audio_dir, augmented_metadata_dir, percentage, augment_type):
    
    type_str = augment_type.strip("type_")
    
    metadata_file = os.path.join(metadata_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_file)

    os.makedirs(augmented_audio_dir, exist_ok=True)
    os.makedirs(augmented_metadata_dir, exist_ok=True)
    
    new_metadata = []

    for class_id, group in metadata.groupby('Class ID'):
        
        files = group['Dataset File Name'].tolist()
        num_files_to_augment = int(len(files) * (percentage / 100.0))
        
        files_to_augment = random.sample(files, num_files_to_augment)
        
        for file in files:
            
            original_audio_file = os.path.join(audio_dir, file)
            shutil.copy(original_audio_file, augmented_audio_dir)
            
            class_name = group[group['Dataset File Name'] == file]['Class Name'].values[0]
            new_metadata.append([file, file, class_id, class_name])
            
            if file in files_to_augment:
                
                if augment_type == "type_A":
                    augmented_file = file.replace('.wav', '_augmented_A.wav')
                    augmented_file_path = os.path.join(augmented_audio_dir, augmented_file)
                    augment_and_save(original_audio_file, augmented_file_path, "type_A")
                    new_metadata.append([augmented_file, augmented_file, class_id, class_name])
                elif augment_type == "type_B":
                    augmented_file = file.replace('.wav', '_augmented_B.wav')
                    augmented_file_path = os.path.join(augmented_audio_dir, augmented_file)
                    augment_and_save(original_audio_file, augmented_file_path, "type_B")
                    new_metadata.append([augmented_file, augmented_file, class_id, class_name])
                elif augment_type == "type_AB":
                    augmented_file = file.replace('.wav', '_augmented_A.wav')
                    augmented_file_path = os.path.join(augmented_audio_dir, augmented_file)
                    augment_and_save(original_audio_file, augmented_file_path, "type_A")
                    new_metadata.append([augmented_file, augmented_file, class_id, class_name])
                    
                    augmented_file = file.replace('.wav', '_augmented_B.wav')
                    augmented_file_path = os.path.join(augmented_audio_dir, augmented_file)
                    augment_and_save(original_audio_file, augmented_file_path, "type_B")
                    new_metadata.append([augmented_file, augmented_file, class_id, class_name])
                else:
                    raise ValueError("Error in type of augmentation.")

    new_metadata_df = pd.DataFrame(new_metadata, columns=['Source File Name', 'Dataset File Name', 'Class ID', 'Class Name'])
    new_metadata_df.to_csv(os.path.join(augmented_metadata_dir, 'metadata.csv'), index=False)


augment_type = "type_A" ## "type_B" ## "type_AB"
tmp = augment_type.strip("type_")
percentage = 50
audio_dir = "../../fsc22_data/raw/audio"
metadata_dir = "../../fsc22_data/raw/metadata"
augmented_audio_dir = f"../../fsc22_data/augmented_{tmp}/audio"
augmented_metadata_dir = f"../../fsc22_data/augmented_{tmp}/metadata"


process_files(audio_dir, metadata_dir, augmented_audio_dir, augmented_metadata_dir, percentage, augment_type)