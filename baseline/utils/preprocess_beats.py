import os
import torch
import torchaudio
import pandas as pd
from beats.BEATs import BEATs, BEATsConfig




def preprocess_audio_get_beatsfeatures(signal, original_sample_rate: int, target_sample_rate: int, num_samples: int, device: str | torch.device):

    signal = signal.to(device)
    signal = resample_if_necessary(signal, original_sample_rate, target_sample_rate, device)
    signal = mix_down_if_necessary(signal)
    signal = cut_if_necessary(signal, num_samples)
    signal = right_pad_if_necessary(signal, num_samples)
    beats_features = get_beats_features(signal, device)

    return beats_features




def preprocess_and_save_data(annotations_dir: str, audio_dir: str, target_sample_rate: int, num_samples: int, device: str | torch.device, output_dir: str):
    
    annotations_file = os.path.join(annotations_dir, "metadata.csv")
    annotations = pd.read_csv(annotations_file)
    
    os.makedirs(output_dir, exist_ok=True)

    for index in range(len(annotations)):
        audio_sample_path = os.path.join(audio_dir, annotations.iloc[index, 1])
        signal, original_sample_rate = torchaudio.load(audio_sample_path)
        melspectrogram = preprocess_audio_get_beatsfeatures(signal, original_sample_rate, target_sample_rate, num_samples, device)
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


def get_beats_features(signal, device):
    
    checkpoint = torch.load('../../pretrained_models/BEATs_iter3_plus_AS2M.pt')

    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()

    beats_features = BEATs_model.extract_features(signal.to("cpu"))[0]
    
    beats_features = beats_features.to(device)
    
    return beats_features





if __name__ == '__main__':
    
    audio_dir = "../../fsc22_data/raw/audio"
    metadata_dir = "../../fsc22_data/raw/metadata"
    output_dir = "../../fsc22_data/preprocessed/beatsfeatures"
    
    TARGET_SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050*5
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    
    preprocess_and_save_data(metadata_dir, audio_dir, TARGET_SAMPLE_RATE, NUM_SAMPLES, DEVICE, output_dir)
