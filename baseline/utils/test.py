import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn


def test(model, data_loader, loss_fn, device, subset_name, num_classes, timestamp):

    softmax = nn.Softmax(dim=1)
    
    model = model.to(device)
    
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    precision = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)

    cum_test_loss = 0.0

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for i, data in enumerate(data_loader):

            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            prob_outputs = softmax(outputs)
            preds = torch.argmax(prob_outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
            loss = loss_fn(outputs, targets)
            cum_test_loss += loss.item()
            
            accuracy.update(preds, targets)
            precision.update(preds, targets)
            recall.update(preds, targets)
            f1_score.update(preds, targets)

    avg_test_loss_per_batch = cum_test_loss / (i+1)
    
    final_accuracy = accuracy.compute()
    final_precision = precision.compute()
    final_recall = recall.compute()
    final_f1_score = f1_score.compute()

    accuracy.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()

    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=[i for i in range(num_classes)], columns=[i for i in range(num_classes)])
    
    cm_df.to_csv(f"logs/fsc22_{timestamp}/metadata/{subset_name}_confusion_matrix.csv", index=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(num_classes)])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    with open(f"logs/fsc22_{timestamp}/metadata/eval_on_{subset_name}_set.txt", 'w') as file:
        file.write(f'Evaluation on {subset_name} set:\n')
        file.write(f'Test Loss: {avg_test_loss_per_batch}\n')
        file.write(f'Accuracy: {final_accuracy}\n')
        file.write(f'Precision: {final_precision}\n')
        file.write(f'Recall: {final_recall}\n')
        file.write(f'F1 Score: {final_f1_score}\n')

    print(f'\nEvaluation on {subset_name} set:\n')
    print(f'Test Loss: {avg_test_loss_per_batch}')
    print(f'Accuracy: {final_accuracy}')
    print(f'Precision: {final_precision}')
    print(f'Recall: {final_recall}')
    print(f'F1 Score: {final_f1_score}')