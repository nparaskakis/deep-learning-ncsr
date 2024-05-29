import os
import pickle
import torch
import torchaudio
import pandas as pd


def preprocess_audio(audio_sample_path: str, target_sample_rate: int, num_samples: int, device: str | torch.device, n_fft: int = 1024, hop_length: int = 512, n_mels: int = 128):
    signal, sr = torchaudio.load(audio_sample_path)

    signal = signal.to(device)
    signal = resample_if_necessary(signal, sr, target_sample_rate, device)
    signal = mix_down_if_necessary(signal)
    signal = cut_if_necessary(signal, num_samples)
    signal = right_pad_if_necessary(signal, num_samples)

    # Convert the signal to a spectrogram
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Not suppoted on mps device
    if device == 'mps':
        spectrogram_transform = spectrogram_transform.to('cpu')
        spectrogram = spectrogram_transform(signal.to('cpu'))
        spectrogram = spectrogram.to(device)
    else:
        spectrogram_transform = spectrogram_transform.to(device)
        spectrogram = spectrogram_transform(signal.to(device))

    return spectrogram


def preprocess_and_save_data(annotations_file: str, audio_dir: str, target_sample_rate: int, num_samples: int, device: str | torch.device, output_dir: str):
    annotations = pd.read_csv(annotations_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index in range(len(annotations)):
        audio_sample_path = os.path.join(audio_dir, annotations.iloc[index, 1])
        spectrogram = preprocess_audio(
            audio_sample_path, target_sample_rate, num_samples, device)

        # Save the spectrogram as a pt file
        output_path = os.path.join(
            output_dir, f"{annotations.iloc[index, 1].rstrip('.wav')}.pt")
        torch.save(spectrogram, output_path)


def resample_if_necessary(signal, sr, target_sample_rate, device):
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sample_rate).to(device)
        signal = resampler(signal)
    return signal


def mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def cut_if_necessary(signal, num_samples):
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    return signal


def right_pad_if_necessary(signal, num_samples):
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal


if __name__ == '__main__':

    # Example usage
    # preprocess_and_save_data('annotations.csv', 'audio_files', 16000, 16000*5, 'cuda', 'output_dir')
    # preprocess and save data
    annotations_file = '../data/raw/metadata/metadata_FSC22.csv'
    audio_dir = '../data/raw/audio'
    target_sample_rate = 22050
    num_samples = 22050*5
    device = torch.device('cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'))
    output_dir = '../data/preprocessed/audio'

    preprocess_and_save_data(annotations_file, audio_dir,
                             target_sample_rate, num_samples, device, output_dir)
