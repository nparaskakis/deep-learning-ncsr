import os

from datetime import datetime

import torchaudio

import argparse

from dataset import *
from training import *
from testing import *
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


from torchviz import make_dot

def write_config_to_file(config, file_path):
    with open(file_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
            
            
def get_model(architecture):
    if architecture == 'model_1':
        from model_1 import CNNNetwork1
        return CNNNetwork1()
    elif architecture == 'model_2':
        from model_2 import CNNNetwork2
        return CNNNetwork2()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
            

def main(args):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    directories = (
        f'logs/fsc22_{timestamp}',
        f'logs/fsc22_{timestamp}/training_losses',
        f'logs/fsc22_{timestamp}/saved_models',
        f'logs/fsc22_{timestamp}/best_model',
        f'logs/fsc22_{timestamp}/metadata'
    )

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Configuration settings
    config = {
        "ANNOTATIONS_FILE": "../data/metadata/metadata_FSC22.csv",
        "AUDIO_DIR": "../data/audio",
        
        "SAMPLE_RATE": 22050,
        "NUM_SAMPLES": 22050*5,
        
        "N_FFT": 1024,
        "HOP_LENGTH": 512,
        "N_MELS": 128,
        
        "BATCH_SIZE": 16,
        "EPOCHS": 2,
        "LEARNING_RATE": 1e-3,
        
        "TRAIN_SIZE": 0.7,
        "VAL_SIZE": 0.15,
        "TEST_SIZE": 0.15,
        
        "MIN_DELTA": 0.001,
        "MIN_LR": 1e-8,
        "SCHEDULER_PATIENCE": 10,
        "EARLY_STOPPING_PATIENCE": 20,
        
        "DEVICE": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    }
    print(f"Using device {config['DEVICE']}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=config["SAMPLE_RATE"],
        n_fft=config["N_FFT"],
        hop_length=config["HOP_LENGTH"],
        n_mels=config["N_MELS"]
    )

    fsc22 = FSC22Dataset(
        annotations_file=config["ANNOTATIONS_FILE"],
        audio_dir=config["AUDIO_DIR"],
        transformation=mel_spectrogram,
        target_sample_rate=config["SAMPLE_RATE"],
        num_samples=config["NUM_SAMPLES"],
        device=config["DEVICE"]
    )
    
    cnn = get_model(args.architecture).to(config["DEVICE"])
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimiser = torch.optim.Adam(
        params=cnn.parameters(),
        lr=config["LEARNING_RATE"]
    )
    
    scheduler = ReduceLROnPlateau(optimiser, 'min', patience=config["SCHEDULER_PATIENCE"], factor=0.1, min_lr=config["MIN_LR"])
    
    train_data_loader, val_data_loader, test_data_loader = get_data_loaders(
        dataset=fsc22,
        train_size=config["TRAIN_SIZE"],
        val_size=config["VAL_SIZE"],
        test_size=config["TEST_SIZE"],
        batch_size=config["BATCH_SIZE"]
    )
    
    x = torch.randn(1, 1, 128, 216).to(config["DEVICE"])
    out = cnn(x)
    dot = make_dot(out, params=dict(list(cnn.named_parameters()) + [('input', x)]))
    dot.render('cnn_architecture', format='png', directory=f'logs/fsc22_{timestamp}/metadata')
    
    model = train(cnn, train_data_loader, val_data_loader, loss_fn, optimiser, scheduler, config["DEVICE"], config["EPOCHS"], timestamp, early_stopping_patience=config["EARLY_STOPPING_PATIENCE"], min_delta=config["MIN_DELTA"])
    
    test(model, test_data_loader, loss_fn, config["DEVICE"], "test", timestamp)
    
    test(model, train_data_loader, loss_fn, config["DEVICE"], "train", timestamp)
    
    test(model, val_data_loader, loss_fn, config["DEVICE"], "val", timestamp)
    
    torch.save(cnn.state_dict(), f'logs/fsc22_{timestamp}/best_model/best_model.pth')
    
    config_file_path = f'logs/fsc22_{timestamp}/metadata/configurations.txt'
    write_config_to_file(config, config_file_path)

    # saved_model = CNNNetwork()
    # saved_model.load_state_dict(torch.load(f'logs/fsc22_{timestamp}/best_model/best_model.pth'))
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio classification with CNNs')
    parser.add_argument('--architecture', type=str, required=True, help='CNN architecture to use')
    args = parser.parse_args()
    main(args)