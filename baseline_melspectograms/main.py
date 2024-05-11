import os

from datetime import datetime

import torchaudio

from model import *
from dataset import *
from training import *
from testing import *

def write_config_to_file(config, file_path):
    with open(file_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
            

if __name__ == "__main__":

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
        "EPOCHS": 20,
        "LEARNING_RATE": 1e-5,
        
        "TRAIN_SIZE": 0.7,
        "VAL_SIZE": 0.15,
        "TEST_SIZE": 0.15,
        
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
        
    cnn = CNNNetwork().to(config["DEVICE"])
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimiser = torch.optim.Adam(
        params=cnn.parameters(),
        lr=config["LEARNING_RATE"]
    )
    
    train_data_loader, val_data_loader, test_data_loader = get_data_loaders(
        dataset=fsc22,
        train_size=config["TRAIN_SIZE"],
        val_size=config["VAL_SIZE"],
        test_size=config["TEST_SIZE"],
        batch_size=config["BATCH_SIZE"]
    )
    
    model = train(cnn, train_data_loader, val_data_loader, loss_fn, optimiser, config["DEVICE"], config["EPOCHS"], timestamp)
    
    test(model, test_data_loader, loss_fn, config["DEVICE"], "test", timestamp)
    
    test(model, train_data_loader, loss_fn, config["DEVICE"], "train", timestamp)
    
    test(model, val_data_loader, loss_fn, config["DEVICE"], "val", timestamp)
    
    torch.save(cnn.state_dict(), f'logs/fsc22_{timestamp}/best_model/best_model.pth')
    
    config_file_path = f'logs/fsc22_{timestamp}/metadata/configurations.txt'
    write_config_to_file(config, config_file_path)
    
    # saved_model = CNNNetwork()
    # saved_model.load_state_dict(torch.load(f'logs/fsc22_{timestamp}/best_model/best_model.pth'))