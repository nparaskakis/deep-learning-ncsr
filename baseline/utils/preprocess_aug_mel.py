import os
import pickle
import torch
import torchaudio
import pandas as pd
import numpy as np
import librosa






def preprocess_audio_get_melspectograms(signal, original_sample_rate: int, target_sample_rate: int, num_samples: int, device: str | torch.device, n_fft: int = 1024, hop_length: int = 512, n_mels: int = 128):

    signal = signal.to(device)
    signal = resample_if_necessary(signal, original_sample_rate, target_sample_rate, device)
    signal = mix_down_if_necessary(signal)
    signal = cut_if_necessary(signal, num_samples)
    signal = right_pad_if_necessary(signal, num_samples)
    melspectrogram = get_melspectrogram(signal, target_sample_rate, n_fft, hop_length, n_mels, device)

    return melspectrogram






def preprocess_and_save_data(annotations_file: str, audio_dir: str, target_sample_rate: int, num_samples: int, device: str | torch.device, output_dir: str, n_fft, hop_length, n_mels):
    
    annotations = pd.read_csv(annotations_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index in range(len(annotations)):
        audio_sample_path = os.path.join(audio_dir, annotations.iloc[index, 1])
        signal, original_sample_rate = torchaudio.load(audio_sample_path)
        melspectrogram = preprocess_audio_get_melspectograms(signal, original_sample_rate, target_sample_rate, num_samples, device, n_fft, hop_length, n_mels)
        output_path = os.path.join(output_dir, f"{annotations.iloc[index, 1].rstrip('.wav')}.pt")
        torch.save(melspectrogram, output_path)




def resample_if_necessary(signal, sr, target_sample_rate, device):
    
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate).to(device)
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


def get_melspectrogram(signal, target_sample_rate, n_fft, hop_length, n_mels, device):
    
    melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=None,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )

    # melspectrogram transformation not suppoted on mps device
    if (device == "mps"):
        melspectrogram_transform = melspectrogram_transform.to('cpu')
        melspectrogram = melspectrogram_transform(signal.to('cpu'))
        melspectrogram = melspectrogram.to(device)
    else:
        melspectrogram_transform = melspectrogram_transform.to(device)
        melspectrogram = melspectrogram_transform(signal.to(device))
        melspectrogram = melspectrogram.to(device)
    
    melspectrogram = librosa.power_to_db(S=melspectrogram.cpu().squeeze(dim=0).numpy(), ref=np.max)
    melspectrogram = torch.tensor(melspectrogram).unsqueeze(0).to(device)
    return melspectrogram






if __name__ == '__main__':
    
    ANNOTATIONS_FILE = '../../data/augmented/metadata/augmented_metadata_FSC22.csv'
    AUDIO_DIR = '../../data/augmented/audio'
    
    TARGET_SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050*5
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
        
    OUTPUT_DIR = '../../data/preprocessed/augmented_melspectrograms'
    
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128

    preprocess_and_save_data(ANNOTATIONS_FILE, AUDIO_DIR, TARGET_SAMPLE_RATE, NUM_SAMPLES, DEVICE, OUTPUT_DIR, N_FFT, HOP_LENGTH, N_MELS)
