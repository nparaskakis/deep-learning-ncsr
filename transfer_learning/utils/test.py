import torch
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn

def test(model, data_loader, loss_fn, device, subset_name, num_classes, timestamp):

    model = model.to(device)
    
    # Multilabel metrics
    accuracy = MultilabelAccuracy(num_labels=num_classes).to(device)
    precision = MultilabelPrecision(num_labels=num_classes, average='macro').to(device)
    recall = MultilabelRecall(num_labels=num_classes, average='macro').to(device)
    f1_score = MultilabelF1Score(num_labels=num_classes, average='macro').to(device)

    cum_test_loss = 0.0

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for i, data in enumerate(data_loader):

            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            prob_outputs = torch.sigmoid(outputs)
            preds = (prob_outputs > 0.5).long()

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

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate multilabel confusion matrices
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    
    for i in range(num_classes):
        cm_df = pd.DataFrame(mcm[i], index=[f'Class {i} Negative', f'Class {i} Positive'], columns=[f'Class {i} Negative', f'Class {i} Positive'])
        cm_df.to_csv(f"logs/fsd50k_{timestamp}/metadata/{subset_name}_confusion_matrix_class_{i}.csv", index=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=mcm[i], display_labels=[f'Class {i} Negative', f'Class {i} Positive'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for Class {i}')
        plt.show()

    with open(f"logs/fsd50k_{timestamp}/metadata/eval_on_{subset_name}_set.txt", 'w') as file:
        file.write(f'Evaluation on {subset_name} set:\n')
        file.write(f'Loss: {avg_test_loss_per_batch}\n')
        file.write(f'Accuracy: {final_accuracy}\n')
        file.write(f'Precision: {final_precision}\n')
        file.write(f'Recall: {final_recall}\n')
        file.write(f'F1 Score: {final_f1_score}\n')

    print(f'\nEvaluation on {subset_name} set:\n')
    print(f'Loss: {avg_test_loss_per_batch}')
    print(f'Accuracy: {final_accuracy}')
    print(f'Precision: {final_precision}')
    print(f'Recall: {final_recall}')
    print(f'F1 Score: {final_f1_score}')
