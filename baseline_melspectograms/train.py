# Import necessary libraries

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader





# Function to train the model for a single epoch

def train_single_epoch(model: torch.nn.Module, train_data_loader: DataLoader, loss_fn: callable, optimiser: torch.optim.Optimizer, device: torch.device | str) -> float:
    
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

def train(model: torch.nn.Module, train_data_loader: DataLoader, val_data_loader: DataLoader, loss_fn: callable, optimiser: torch.optim.Optimizer, device: torch.device | str, epochs: int):
    
    # Timestamp for logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # TensorBoard writer instance
    writer = SummaryWriter(f'runs/fsc22_{timestamp}')
    
    # Initialize best validation loss for model saving
    best_val_loss = np.inf
    
    # Training loop
    for epoch_number in range(epochs):
        
        # Set the model to training mode
        model.train()
        
        print(f"Epoch {epoch_number+1}")
        
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
        print("---------------------------")
        
        writer.add_scalars(
            'Training vs. Validation Loss', {
                'Training': avg_train_loss_per_batch,
                'Validation': avg_val_loss_per_batch
            },
            epoch_number + 1
        )
        
        writer.flush()
        
        # Save model if validation loss improved
        if avg_val_loss_per_batch < best_val_loss:
            best_val_loss = avg_val_loss_per_batch
            model_path = f'model_{timestamp}_{epoch_number}'
            torch.save(model.state_dict(), model_path)
    
    print("Finished training")
    
    return model