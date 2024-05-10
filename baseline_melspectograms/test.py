# Import necessary libraries

import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score





# Function for evaluating the model on the test set

def test(model, test_data_loader, loss_fn, device):
    
    # Initialize metric calculators for accuracy, precision, recall, and F1 score.
    accuracy = Accuracy(task="multiclass", num_classes=27).to(device)
    precision = Precision(task="multiclass", num_classes=27, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=27, average='macro').to(device)
    f1_score = F1Score(task="multiclass", num_classes=27, average='macro').to(device)

    # Cumulative loss for the test dataset
    cum_test_loss = 0
    
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation for inference
    with torch.no_grad():
        
        for i, data in enumerate(test_data_loader):
            
            # Unpack the batch of test data
            inputs, targets = data
            
            # Move data to the specified device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass: compute the model's outputs
            outputs = model(inputs)
            
            # Get the predicted classes
            preds = torch.argmax(outputs, dim=1)
            
            # Compute the loss between outputs and targets
            loss = loss_fn(outputs, targets)
            
            # Accumulate the batch loss
            cum_test_loss += loss.item()
            
            # Update metric calculators
            accuracy.update(preds, targets)
            precision.update(preds, targets)
            recall.update(preds, targets)
            f1_score.update(preds, targets)

    # Compute average loss and final metric values
    avg_test_loss_per_batch = cum_test_loss / (i + 1)
    final_accuracy = accuracy.compute()
    final_precision = precision.compute()
    final_recall = recall.compute()
    final_f1_score = f1_score.compute()

    # Reset metrics for future use
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()

    # Print the computed metrics
    print(f'AVG Loss: {avg_test_loss_per_batch}')
    print(f'Accuracy: {final_accuracy}')
    print(f'Precision: {final_precision}')
    print(f'Recall: {final_recall}')
    print(f'F1 Score: {final_f1_score}')
    
    return