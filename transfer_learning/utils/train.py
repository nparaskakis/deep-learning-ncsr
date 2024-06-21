import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score

def train_single_epoch(model: torch.nn.Module, train_data_loader: DataLoader, loss_fn: callable, optimizer: torch.optim.Optimizer,  device: torch.device | str) -> float:

    model.train()
    
    cum_train_loss = 0.0
    
    for i, data in enumerate(train_data_loader):
        input, target = data
        input, target = input.to(device), target.to(device)
        prediction = model(input)
        loss = loss_fn(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_train_loss += loss.item()

    avg_train_loss_per_batch = cum_train_loss / (i+1)

    return avg_train_loss_per_batch




def validate(model: torch.nn.Module, val_data_loader: DataLoader, loss_fn: callable, num_classes, device: torch.device | str) -> float:

    # softmax = nn.Softmax(dim=1).to(device)
    accuracy = MultilabelAccuracy(num_labels=num_classes).to(device)
    precision = MultilabelPrecision(num_labels=num_classes, average='macro').to(device)
    recall = MultilabelRecall(num_labels=num_classes, average='macro').to(device)
    f1_score = MultilabelF1Score(num_labels=num_classes, average='macro').to(device)
    
    model.eval()

    cum_val_loss = 0.0

    with torch.no_grad():

        for i, data in enumerate(val_data_loader):
            input, target = data
            input, target = input.to(device), target.to(device)
            outputs = model(input)
            prob_outputs = torch.sigmoid(outputs)
            preds = (prob_outputs > 0.5)
            loss = loss_fn(outputs, target)
            cum_val_loss += loss.item()
            f1_score.update(preds, target)

    avg_val_loss_per_batch = cum_val_loss / (i+1)
    
    final_f1_score = f1_score.compute()
    f1_score.reset()
    
    return avg_val_loss_per_batch, final_f1_score





def train(model: torch.nn.Module, train_data_loader: DataLoader, val_data_loader: DataLoader, loss_fn: callable, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, device: torch.device | str, epochs: int, num_classes, timestamp, early_stopping_patience, min_delta):

    writer = SummaryWriter(f'logs/fsd50k_{timestamp}/training_losses')

    best_val_loss = np.inf
    best_model = model
    
    early_stopping_counter = 0

    with open(f'logs/fsd50k_{timestamp}/metadata/training_log.txt', 'w') as log_file:

        for epoch_number in range(epochs):

            print(f"Epoch {epoch_number+1}")
            print(f"Epoch {epoch_number+1}", file=log_file)

            avg_train_loss_per_batch = train_single_epoch(model, train_data_loader, loss_fn, optimizer, device)
            avg_val_loss_per_batch, val_f1_score = validate(model, val_data_loader, loss_fn, num_classes, device)

            print(f"Train Loss: {avg_train_loss_per_batch} Val Loss {avg_val_loss_per_batch} Val f1_score {val_f1_score}")
            print(f"Train Loss {avg_train_loss_per_batch} Val Loss {avg_val_loss_per_batch} Val f1_score {val_f1_score}", file=log_file)
            print("---------------------------")
            print("---------------------------", file=log_file)

            writer.add_scalars(
                'Training_vs_Validation_Loss', {
                    'Training': avg_train_loss_per_batch,
                    'Validation': avg_val_loss_per_batch
                },
                epoch_number + 1
            )
            
            writer.add_scalars(
                'Val_f1_score', {
                    'Val_f1_score': val_f1_score
                },
                epoch_number + 1
            )

            writer.flush()

            lrs_before_step = [group['lr'] for group in optimizer.param_groups]

            scheduler.step(avg_val_loss_per_batch)

            lrs_after_step = [group['lr'] for group in optimizer.param_groups]

            for lr_before, lr_after in zip(lrs_before_step, lrs_after_step):
                if lr_after < lr_before:
                    print(f"Learning rate decreased from {lr_before} to {lr_after}")
                    print(f"Learning rate decreased from {lr_before} to {lr_after}", file=log_file)

            if (best_val_loss - avg_val_loss_per_batch > min_delta):
                best_val_loss = avg_val_loss_per_batch
                early_stopping_counter = 0
                best_model = model
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    print("Finished training")

    torch.save(best_model.state_dict(), f"logs/fsd50k_{timestamp}/best_model/model.pth")

    return best_model
