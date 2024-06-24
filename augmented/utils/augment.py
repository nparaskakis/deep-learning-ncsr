import os
import pandas as pd
import shutil
import random
from audiomentations import Compose, TimeMask, Gain, TimeStretch, Shift, AddGaussianNoise, PitchShift
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split


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
        
    
    

def process_files(audio_dir, metadata_dir, augmented_audio_dir, augmented_metadata_dir, percentage, augment_type, train_size, val_size, test_size, random_seed=42):
    
    metadata_file = os.path.join(metadata_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_file)
    
    assert train_size + val_size + test_size == 1, "The sum of sizes must be 1."

    train_val, test_data = train_test_split(metadata, test_size=test_size, stratify=metadata['Class ID'], random_state=random_seed)
    train_data, val_data = train_test_split(train_val, test_size=val_size/(train_size + val_size), stratify=train_val['Class ID'], random_state=random_seed)

    train_data['split'] = 'train'
    val_data['split'] = 'val'
    test_data['split'] = 'test'

    combined_metadata = pd.concat([train_data, val_data, test_data])

    # combined_metadata.to_csv(os.path.join(metadata_dir, 'metadata_with_split.csv'), index=False)
    
    os.makedirs(augmented_audio_dir, exist_ok=True)
    os.makedirs(augmented_metadata_dir, exist_ok=True)
    
    new_metadata = []

    for class_id, group in train_data.groupby('Class ID'):
        
        files = group['Dataset File Name'].tolist()
        num_files_to_augment = int(len(files) * (percentage / 100.0))
        
        files_to_augment = random.sample(files, num_files_to_augment)
        
        for file in files:
            
            original_audio_file = os.path.join(audio_dir, file)
            shutil.copy(original_audio_file, augmented_audio_dir)
            
            class_name = group[group['Dataset File Name'] == file]['Class Name'].values[0]
            new_metadata.append([file, file, class_id, class_name, 'train'])
            
            if file in files_to_augment:
                
                if augment_type == "type_A":
                    augmented_file = file.replace('.wav', '_augmented_A.wav')
                    augmented_file_path = os.path.join(augmented_audio_dir, augmented_file)
                    augment_and_save(original_audio_file, augmented_file_path, "type_A")
                    new_metadata.append([augmented_file, augmented_file, class_id, class_name, 'train'])
                elif augment_type == "type_B":
                    augmented_file = file.replace('.wav', '_augmented_B.wav')
                    augmented_file_path = os.path.join(augmented_audio_dir, augmented_file)
                    augment_and_save(original_audio_file, augmented_file_path, "type_B")
                    new_metadata.append([augmented_file, augmented_file, class_id, class_name, 'train'])
                elif augment_type == "type_AB":
                    augmented_file = file.replace('.wav', '_augmented_A.wav')
                    augmented_file_path = os.path.join(augmented_audio_dir, augmented_file)
                    augment_and_save(original_audio_file, augmented_file_path, "type_A")
                    new_metadata.append([augmented_file, augmented_file, class_id, class_name, 'train'])
                    
                    augmented_file = file.replace('.wav', '_augmented_B.wav')
                    augmented_file_path = os.path.join(augmented_audio_dir, augmented_file)
                    augment_and_save(original_audio_file, augmented_file_path, "type_B")
                    new_metadata.append([augmented_file, augmented_file, class_id, class_name, 'train'])
                else:
                    raise ValueError("Error in type of augmentation.")
    
    val_test_metadata = combined_metadata[combined_metadata['split'] != 'train']
    for index, row in val_test_metadata.iterrows():
        original_audio_file = os.path.join(audio_dir, row['Dataset File Name'])
        shutil.copy(original_audio_file, augmented_audio_dir)
        new_metadata.append([row['Dataset File Name'], row['Dataset File Name'], row['Class ID'], row['Class Name'], row['split']])

    new_metadata_df = pd.DataFrame(new_metadata, columns=['Source File Name', 'Dataset File Name', 'Class ID', 'Class Name', 'split'])
    new_metadata_df.to_csv(os.path.join(augmented_metadata_dir, 'metadata.csv'), index=False)


augment_type = "type_A" ## "type_B" ## "type_AB"
tmp = augment_type.strip("type_")
percentage = 100
audio_dir = "../../fsc22_data/raw/audio"
metadata_dir = "../../fsc22_data/raw/metadata"
augmented_audio_dir = f"../../fsc22_data/augmented_{tmp}_{percentage}/audio"
augmented_metadata_dir = f"../../fsc22_data/augmented_{tmp}_{percentage}/metadata"


process_files(audio_dir, metadata_dir, augmented_audio_dir, augmented_metadata_dir, percentage, augment_type, 0.7, 0.15, 0.15)