import os
from datetime import datetime
import argparse
from utils.dataset import FSC22Dataset, get_data_loaders
from utils.train import *
from utils.test import *
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


from torchviz import make_dot


def write_config_to_file(config, file_path):
    with open(file_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def get_model(architecture, dim1, dim2, num_classes):
    if architecture == 'CNN1':
        from models.cnn1 import CNNNetwork1
        return CNNNetwork1(dim1, dim2, num_classes)
    elif architecture == 'CNN2':
        from models.cnn2 import CNNNetwork2
        return CNNNetwork2(dim1, dim2, num_classes)
    elif architecture == 'FCNN1':
        from models.fcnn1 import FCNNNetwork1
        return FCNNNetwork1(dim1, dim2, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")






def validate_arguments(args):
    
    if (args.architecture != "CNN1") and \
       (args.architecture != "CNN2") and \
       (args.architecture != "FCNN1"):
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
    if (args.features != "melspectrograms") and \
       (args.features != "audiofeatures") and \
       (args.features != "beatsfeatures") and \
       (args.features != "augmented_A_melspectrograms") and \
       (args.features != "augmented_B_melspectrograms") and \
       (args.features != "augmented_AB_melspectrograms") and \
       (args.features != "augmented_A_audiofeatures") and \
       (args.features != "augmented_B_audiofeatures") and \
       (args.features != "augmented_AB_audiofeatures"): 
        raise ValueError(f"Unknown features: {args.features}")
    
    
    
    


def main(args):

    validate_arguments(args)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    directories = (
        f'logs/fsc22_{timestamp}',
        f'logs/fsc22_{timestamp}/training_losses',
        f'logs/fsc22_{timestamp}/best_model',
        f'logs/fsc22_{timestamp}/metadata'
    )

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    config = {
        # "ANNOTATIONS_FILE": "../data/raw/metadata/metadata.csv",
        "ANNOTATIONS_FILE": "../data/augmented/metadata/metadata.csv",
        "AUDIO_DIR": f"../data/preprocessed/{args.features}",

        "BATCH_SIZE": 128,
        "EPOCHS": 50,
        "LEARNING_RATE": 1e-5,

        "TRAIN_SIZE": 0.7,
        "VAL_SIZE": 0.15,
        "TEST_SIZE": 0.15,

        "MIN_DELTA": 0.001,
        "MIN_LR": 1e-8,
        
        "SCHEDULER_PATIENCE": 10,
        "EARLY_STOPPING_PATIENCE": 25,
        
        "NUM_CLASSES": 27,

        "DEVICE": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    }
    
    print(f"Using device {config['DEVICE']}")

    fsc22 = FSC22Dataset(annotations_file=config["ANNOTATIONS_FILE"], data_dir=config["AUDIO_DIR"], device=config["DEVICE"])
    dim1 = fsc22[0][0].shape[1]
    dim2 = fsc22[0][0].shape[2]
    
    cnn = get_model(args.architecture, dim1, dim2, config["NUM_CLASSES"]).to(config["DEVICE"])

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=cnn.parameters(), lr=config["LEARNING_RATE"])

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config["SCHEDULER_PATIENCE"], factor=0.1, min_lr=config["MIN_LR"])

    train_data_loader, val_data_loader, test_data_loader = get_data_loaders(
        dataset=fsc22,
        train_size=config["TRAIN_SIZE"],
        val_size=config["VAL_SIZE"],
        test_size=config["TEST_SIZE"],
        batch_size=config["BATCH_SIZE"]
    )

    x = torch.randn(1, 1, dim1, dim2).to(config["DEVICE"])
    out = cnn(x)
    dot = make_dot(out, params=dict(list(cnn.named_parameters()) + [('input', x)]))
    dot.render('cnn_architecture', format='png', directory=f'logs/fsc22_{timestamp}/metadata')

    model = train(cnn, train_data_loader, val_data_loader, loss_fn, optimizer, scheduler, config["DEVICE"], config["EPOCHS"], 27, timestamp, early_stopping_patience=config["EARLY_STOPPING_PATIENCE"], min_delta=config["MIN_DELTA"])

    test(model, train_data_loader, loss_fn, config["DEVICE"], "train", config["NUM_CLASSES"], timestamp)
    test(model, val_data_loader, loss_fn, config["DEVICE"], "val", config["NUM_CLASSES"], timestamp)
    test(model, test_data_loader, loss_fn, config["DEVICE"], "test", config["NUM_CLASSES"], timestamp) 

    config_file_path = f'logs/fsc22_{timestamp}/metadata/configurations.txt'
    write_config_to_file(config, config_file_path)

    # saved_model = CNNNetwork()
    # saved_model.load_state_dict(torch.load(f'logs/fsc22_{timestamp}/best_model/best_model.pth'))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Forest Sounds Classification')
    parser.add_argument('--architecture', type=str, required=True, help='CNN architecture to use')
    parser.add_argument('--features', type=str, required=True, help='Features to use (melspectrograms or audiofeatures)')
    args = parser.parse_args()
    
    main(args)