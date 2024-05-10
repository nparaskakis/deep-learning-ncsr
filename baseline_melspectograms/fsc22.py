import torchaudio
from dataset import *



if __name__ == "__main__":
    
    ANNOTATIONS_FILE = "data/metadata/metadata_FSC22.csv"
    AUDIO_DIR = "data/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050*5
    
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 1e-6

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )

    fsc22 = FSC22Dataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectrogram,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device=device
    )
    
    print(f"There are {len(fsc22)} samples in the dataset.")
    
    cnn = CNNNetwork().to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimiser = torch.optim.Adam(
        params=cnn.parameters(),
        lr=LEARNING_RATE
    )
    
    train_data_loader, val_data_loader, test_data_loader = get_data_loaders(
        dataset=fsc22,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        batch_size=BATCH_SIZE
    )
    
    model = train(cnn, train_data_loader, val_data_loader, loss_fn, optimiser, device, EPOCHS)
    
    test(model, test_data_loader, loss_fn, device)
    
    torch.save(cnn.state_dict(), "final_model.pth")
    print("Trained feed forward net saved at final_model.pth")
    
    saved_model = CNNNetwork()
    saved_model.load_state_dict(torch.load('final_model.pth'))