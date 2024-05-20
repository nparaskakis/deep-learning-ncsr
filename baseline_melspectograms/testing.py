# Import necessary libraries

import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np




# Function for evaluating the model on the test set

def test(model, data_loader, loss_fn, device, subset_name, timestamp):
    
    model = model.to(device)
    # Initialize metric calculators for accuracy, precision, recall, and F1 score.
    accuracy = Accuracy(task="multiclass", num_classes=27).to(device)
    precision = Precision(task="multiclass", num_classes=27, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=27, average='macro').to(device)
    f1_score = F1Score(task="multiclass", num_classes=27, average='macro').to(device)

    # Cumulative loss for the test dataset
    cum_test_loss = 0
    
    # Set the model to evaluation mode
    model.eval()

    all_preds = []
    all_labels = []

    # Disable gradient computation for inference
    with torch.no_grad():
        
        for i, data in enumerate(data_loader):
            
            # Unpack the batch of test data
            inputs, targets = data
            
            # Move data to the specified device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass: compute the model's outputs
            outputs = model(inputs)
            
            # Get the predicted classes
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
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
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(27)])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # Open a file and write the results
    with open(f'logs/fsc22_{timestamp}/metadata/eval_on_{subset_name}_set.txt', 'w') as file:
        file.write(f'Evaluation on {subset_name} set:\n')
        file.write(f'AVG Loss: {avg_test_loss_per_batch}\n')
        file.write(f'Accuracy: {final_accuracy}\n')
        file.write(f'Precision: {final_precision}\n')
        file.write(f'Recall: {final_recall}\n')
        file.write(f'F1 Score: {final_f1_score}\n')
    
    # Print the computed metrics
    print(f'\nEvaluation on {subset_name} set:\n')
    print(f'AVG Loss: {avg_test_loss_per_batch}')
    print(f'Accuracy: {final_accuracy}')
    print(f'Precision: {final_precision}')
    print(f'Recall: {final_recall}')
    print(f'F1 Score: {final_f1_score}')
    
    return