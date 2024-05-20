# Import necessary libraries

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter




# Function to train the model for a single epoch

def train_single_epoch(model: torch.nn.Module, train_data_loader: DataLoader, loss_fn: callable, optimiser: torch.optim.Optimizer,  device: torch.device | str) -> float:
    
    # Cumulative loss for the epoch
    cum_train_loss = 0.0
    
    # Loop through the batches of data in the DataLoader
    for i, data in enumerate(train_data_loader):
        
        # Unpack the data. 'input' is the feature, 'target' is the true label.
        input, target = data
        
        # Move the data to the specified device
        input, target = input.to(device), target.to(device)
        
        # Forward pass: Compute predicted output by passing input to the model
        prediction = model(input)
        
        # Calculate the loss
        loss = loss_fn(prediction, target)
        
        # Zero the gradients before running the backward pass.
        optimiser.zero_grad()
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Update model parameters
        optimiser.step()
        
        # Update cumulative loss
        cum_train_loss += loss.item()

    # Calculate average loss per batch
    avg_train_loss_per_batch = cum_train_loss / (i+1)
    
    return avg_train_loss_per_batch





# Function to train the model along with validating its performance (monitoring the validation loss)

def train(model: torch.nn.Module, train_data_loader: DataLoader, val_data_loader: DataLoader, loss_fn: callable, optimiser: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, device: torch.device | str, epochs: int, timestamp, early_stopping_patience, min_delta):
    
    # TensorBoard writer instance
    writer = SummaryWriter(f'logs/fsc22_{timestamp}/training_losses')
    
    # Initialize best validation loss for model saving
    best_val_loss = np.inf
    best_model_path = ""
    best_model = model
    early_stopping_counter = 0

    # Open a text file for output
    with open(f'logs/fsc22_{timestamp}/metadata/training_log.txt', 'w') as log_file:
        
        # Training loop
        for epoch_number in range(epochs):
            
            # Set the model to training mode
            model.train()
            
            # Print to both console and log file
            print(f"Epoch {epoch_number+1}")
            print(f"Epoch {epoch_number+1}", file=log_file)
            
            # Train for a single epoch and return average loss per batch
            avg_train_loss_per_batch = train_single_epoch(model, train_data_loader, loss_fn, optimiser, device)
            
            # Set the model to evaluation mode
            model.eval()

            # Cumulative validation loss
            cum_val_loss = 0.0
            
            # Inference mode, gradients not needed
            with torch.no_grad():
            
                # Loop through the batches of data in the DataLoader
                for i, data in enumerate(val_data_loader):
                    
                    # Unpack the data. 'input' is the feature, 'target' is the true label.
                    input, target = data
                    
                    # Move the data to the specified device
                    input, target = input.to(device), target.to(device)
                    
                    # Forward pass
                    outputs = model(input)
                    
                    # Calculate loss
                    loss = loss_fn(outputs, target)
                    
                    # Update cumulative loss
                    cum_val_loss += loss.item()
            
            # Average validation loss per batch
            avg_val_loss_per_batch = cum_val_loss / (i + 1)
            
            # Log training and validation loss
            print(f"LOSS train {avg_train_loss_per_batch} valid {avg_val_loss_per_batch}")
            print(f"LOSS train {avg_train_loss_per_batch} valid {avg_val_loss_per_batch}", file=log_file)
            print("---------------------------")
            print("---------------------------", file=log_file)
            
            writer.add_scalars(
                'Training vs. Validation Loss', {
                    'Training': avg_train_loss_per_batch,
                    'Validation': avg_val_loss_per_batch
                },
                epoch_number + 1
            )
            
            writer.flush()
            
            scheduler.step(avg_val_loss_per_batch)
            
            # Save model if validation loss improved
            if  (best_val_loss - avg_val_loss_per_batch > min_delta):
                best_val_loss = avg_val_loss_per_batch
                early_stopping_counter = 0
                best_model_path = f'logs/fsc22_{timestamp}/saved_models/model_{epoch_number+1}.pth'
                best_model = model
                torch.save(model.state_dict(), best_model_path)
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    
    print("Finished training")
    
    best_model.load_state_dict(torch.load(best_model_path))
    
    return best_model