import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import librosa



def preprocess_audio_get_audio_features(signal, original_sample_rate: int, target_sample_rate: int, num_samples: int, device: str | torch.device, n_fft: int = 2048, hop_length: int = 512):

    signal = signal.to(device)
    signal = resample_if_necessary(signal, original_sample_rate, target_sample_rate, device)
    signal = mix_down_if_necessary(signal)
    signal = cut_if_necessary(signal, num_samples)
    signal = right_pad_if_necessary(signal, num_samples)
    melspectrogram = get_audio_features(signal, target_sample_rate, n_fft, hop_length, device)

    return melspectrogram




def preprocess_and_save_data(annotations_dir: str, audio_dir: str, target_sample_rate: int, num_samples: int, device: str | torch.device, output_dir: str, n_fft, hop_length):
    
    annotations_file = os.path.join(annotations_dir, "metadata.csv")
    annotations = pd.read_csv(annotations_file)
    
    os.makedirs(output_dir, exist_ok=True)

    for index in range(len(annotations)):
        audio_sample_path = os.path.join(audio_dir, annotations.iloc[index, 0])
        signal, original_sample_rate = torchaudio.load(audio_sample_path)
        melspectrogram = preprocess_audio_get_audio_features(signal, original_sample_rate, target_sample_rate, num_samples, device, n_fft, hop_length)
        output_path = os.path.join(output_dir, f"{annotations.iloc[index, 0].rstrip('.wav')}.pt")
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



def get_audio_features(signal, target_sample_rate, n_fft, hop_length, device):
    
    signal = signal.cpu().numpy()

    mfccs = librosa.feature.mfcc(y=signal, sr=target_sample_rate, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
    chroma_stft = librosa.feature.chroma_stft(y=signal, sr=target_sample_rate, n_chroma=12, n_fft=n_fft, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=target_sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=target_sample_rate, n_fft=n_fft, hop_length=hop_length, n_bands=6)
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=target_sample_rate, n_fft=n_fft, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=target_sample_rate, n_fft=n_fft, hop_length=hop_length)
    spectral_flatness = librosa.feature.spectral_flatness(y=signal, n_fft=n_fft, hop_length=hop_length, power=2.0)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=target_sample_rate, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=n_fft, hop_length=hop_length)
    
    # Concatenate features row-wise
    features_tensor = np.concatenate((mfccs, chroma_stft, tonnetz, spectral_contrast, 
                                spectral_centroids, spectral_bandwidth, spectral_flatness, 
                                spectral_rolloff, zero_crossing_rate), axis=1)
    
    features_tensor = torch.tensor(features_tensor, dtype=torch.float32).to(device)
    
    return features_tensor




if __name__ == '__main__':
    
    audio_dir = "../../urb_data/raw/audio"
    metadata_dir = "../../urb_data/raw/metadata"
    output_dir = "../../urb_data/preprocessed/audiofeatures"
    
    TARGET_SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050*5
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
        
    N_FFT = 2048
    HOP_LENGTH = 512

    preprocess_and_save_data(metadata_dir, audio_dir, TARGET_SAMPLE_RATE, NUM_SAMPLES, DEVICE, output_dir, N_FFT, HOP_LENGTH)
