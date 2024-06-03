import os
import shutil
import pandas as pd
import librosa as lr
import soundfile as sf
from collections import Counter
from audiomentations import Compose, TimeMask, Gain, TimeStretch, Shift


def augment_and_save_data(audio_dir: str, metadata_file: str, output_dir: str, percentage_increase: float):
    augment = Compose([
        # AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.02, p=1),
        # PitchShift(min_semitones=-4, max_semitones=4, p=1),
        # HighPassFilter(min_cutoff_freq=1500, max_cutoff_freq=2500, p=1),
        TimeMask(min_band_part=0.1, max_band_part=0.5, p=0.5),
        Gain(min_gain_db=-12, max_gain_db=12, p=1),
        TimeStretch(min_rate=0.9, max_rate=1.2, p=0.8),
        Shift(min_shift=-0.5, max_shift=0.5, rollover=True, p=1),
        # ClippingDistortion(min_percentile_threshold=0,
        #    max_percentile_threshold=40, p=1),
        # LowPassFilter(min_cutoff_freq=1500, max_cutoff_freq=8000, p=1),
        # PolarityInversion(p=1)
    ])

    annotations = pd.read_csv(metadata_file)
    class_counts = Counter(annotations['Class ID'])

    total_samples_per_class = {}
    for class_id, count in class_counts.items():
        num_samples_increase = int(count * percentage_increase)
        total_samples_per_class[class_id] = count + num_samples_increase

    augmented_audio_dir = os.path.join(output_dir, 'audio')
    os.makedirs(augmented_audio_dir, exist_ok=True)

    augmented_annotations = []

    # Copy original files to output directory
    for index, row in annotations.iterrows():
        original_audio_path = os.path.join(audio_dir, row['Dataset File Name'])
        output_audio_path = os.path.join(
            augmented_audio_dir, row['Dataset File Name'])
        shutil.copyfile(original_audio_path, output_audio_path)
        augmented_annotations.append(row)

    for class_id, count in class_counts.items():
        # Select random samples from each class
        class_samples = annotations[annotations['Class ID'] == class_id]
        if len(class_samples) > 0:
            num_samples_to_generate = total_samples_per_class[class_id] - \
                class_counts[class_id]
            selected_samples = class_samples.sample(n=min(
                num_samples_to_generate, len(class_samples)), replace=False)
        else:
            continue

        for index, row in selected_samples.iterrows():
            original_audio_path = os.path.join(
                audio_dir, row['Dataset File Name'])
            signal, sr = lr.load(original_audio_path)

            augmented_audio = augment(signal, sample_rate=sr)

            output_filename = f"{row['Dataset File Name'][:-4]}_augmented.wav"
            output_path = os.path.join(augmented_audio_dir, output_filename)

            sf.write(output_path, augmented_audio, sr)

            # Append augmented data
            new_row = row.copy()
            new_row['Dataset File Name'] = output_filename
            augmented_annotations.append(new_row)

    augmented_annotations_df = pd.DataFrame(augmented_annotations)
    augmented_metadata_file = os.path.join(
        output_dir, 'augmented_metadata.csv')
    augmented_annotations_df.to_csv(augmented_metadata_file, index=False)


if __name__ == "__main__":
    audio_dir = "../data/raw/audio"
    metadata_file = "../data/raw/metadata/metadata_FSC22.csv"
    output_dir = "../data/augmented"
    percentage_increase = 0.1  # 10% increase

    augment_and_save_data(audio_dir, metadata_file,
                          output_dir, percentage_increase)
