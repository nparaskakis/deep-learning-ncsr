import os
from datetime import datetime
import argparse
from utils.dataset import FSD50KDataset, get_data_loaders
from utils.train import *
from utils.test import *
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models

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
    elif architecture == 'CNN3':
        from models.cnn3 import CNNNetwork3
        return CNNNetwork3(dim1, dim2, num_classes)
    elif architecture == 'CNN4':
        from models.cnn4 import CNNNetwork4
        return CNNNetwork4(dim1, dim2, num_classes)
    elif architecture == 'CNN5':
        from models.cnn5 import CNNNetwork5
        return CNNNetwork5(dim1, dim2, num_classes)
    elif architecture == 'FCNN1':
        from models.fcnn1 import FCNNNetwork1
        return FCNNNetwork1(dim1, dim2, num_classes)
    elif architecture == 'MobileNetV3Small':
        mobilenet_v3_small = models.mobilenet_v3_small(weights=None)
        mobilenet_v3_small.classifier[3] = torch.nn.Linear(mobilenet_v3_small.classifier[3].in_features, num_classes)
        return mobilenet_v3_small
    else:
        raise ValueError(f"Unknown architecture: {architecture}")






def validate_arguments(args):
    
    if (args.architecture != "CNN1") and \
       (args.architecture != "CNN2") and \
       (args.architecture != "CNN3") and \
       (args.architecture != "CNN4") and \
       (args.architecture != "CNN5") and \
       (args.architecture != "FCNN1") and \
       (args.architecture != "MobileNetV3Small"):
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
    if (args.features != "melspectrograms") and \
       (args.features != "audiofeatures"):
        raise ValueError(f"Unknown features: {args.features}")
    
    
    
    


def main(args):

    validate_arguments(args)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    directories = (
        f'logs/fsd50k_{timestamp}',
        f'logs/fsd50k_{timestamp}/training_losses',
        f'logs/fsd50k_{timestamp}/best_model',
        f'logs/fsd50k_{timestamp}/metadata'
    )

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    config = {
        "ANNOTATIONS_FILE": "../fsd50k_data/raw/metadata/metadata.csv",
        "VOCABULARY_FILE": "../fsd50k_data/raw/metadata/vocabulary.csv",
        "AUDIO_DIR": f"../fsd50k_data/preprocessed/{args.features}",

        "BATCH_SIZE": 128,
        "EPOCHS": int(args.epochs) if args.epochs else 100,
        "LEARNING_RATE": 1e-5,

        "MIN_DELTA": 0.001,
        "MIN_LR": 1e-8,
        
        "SCHEDULER_PATIENCE": 10,
        "EARLY_STOPPING_PATIENCE": 25,
        
        "NUM_CLASSES": 200,

        "DEVICE": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        
        "CONTLEARN": args.contlearn
    }
    
    print(f"Using device {config['DEVICE']}")

    config_file_path = f'logs/fsd50k_{timestamp}/metadata/configurations.txt'
    write_config_to_file(config, config_file_path)
    
    fsd50k = FSD50KDataset(annotations_file=config["ANNOTATIONS_FILE"], vocabulary_file=config["ANNOTATIONS_FILE"], data_dir=config["AUDIO_DIR"], device=config["DEVICE"], model_str="mobilenet")
    dim1 = fsd50k[0][0].shape[1]
    dim2 = fsd50k[0][0].shape[2]
    
    cnn = get_model(args.architecture, dim1, dim2, config["NUM_CLASSES"]).to(config["DEVICE"])
    
    if args.contlearn != None:
        cnn.load_state_dict(torch.load(args.contlearn))
        print("Trained model loaded.")
    
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(params=cnn.parameters(), lr=config["LEARNING_RATE"])

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config["SCHEDULER_PATIENCE"], factor=0.1, min_lr=config["MIN_LR"])

    train_data_loader, val_data_loader, test_data_loader = get_data_loaders(
        dataset=fsd50k,
        batch_size=config["BATCH_SIZE"]
    )

    x = torch.randn(1, 1, dim1, dim2).to(config["DEVICE"])
    x = x.repeat(1, 3, 1, 1)
    out = cnn(x)
    dot = make_dot(out, params=dict(list(cnn.named_parameters()) + [('input', x)]))
    dot.render('cnn_architecture', format='png', directory=f'logs/fsd50k_{timestamp}/metadata')

    model = train(cnn, train_data_loader, val_data_loader, loss_fn, optimizer, scheduler, config["DEVICE"], config["EPOCHS"], config["NUM_CLASSES"], timestamp, early_stopping_patience=config["EARLY_STOPPING_PATIENCE"], min_delta=config["MIN_DELTA"])
    
    test(model, train_data_loader, loss_fn, config["DEVICE"], "train", config["NUM_CLASSES"], timestamp)
    test(model, val_data_loader, loss_fn, config["DEVICE"], "val", config["NUM_CLASSES"], timestamp)
    test(model, test_data_loader, loss_fn, config["DEVICE"], "test", config["NUM_CLASSES"], timestamp) 

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Forest Sounds Classification')
    parser.add_argument('--architecture', type=str, required=True, help='CNN architecture to use')
    parser.add_argument('--features', type=str, required=True, help='Features to use (melspectrograms or audiofeatures)')
    parser.add_argument('--contlearn', type=str, help='Continue learning of a trained model')
    parser.add_argument('--epochs', type=str, help='Epochs to run')
    args = parser.parse_args()
    
    main(args)