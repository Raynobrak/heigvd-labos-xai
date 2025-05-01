# evaluate.py
import os
import torch
import argparse
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from scripts.train import get_device
from torch.utils.data import DataLoader
from scripts.helpers import get_datasets
from efficientnet_pytorch import EfficientNet

def evaluate(model, batch_size):
    device = get_device()    
    # prepare test loader
    test_dataset = get_datasets(train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_true, all_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            if labels.ndim > 1:
                labels = torch.argmax(labels, dim=1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    # define full label set to avoid warnings
    labels_list = list(range(8))
    # compute metrics
    total = len(all_true)
    correct = sum(int(t == p) for t, p in zip(all_true, all_pred))
    accuracy = correct / total
    f1 = f1_score(all_true, all_pred, average='macro', labels=labels_list)
    cm = confusion_matrix(all_true, all_pred, labels=labels_list)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return f1, accuracy, cm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the .pth model file')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    args = parser.parse_args()
    
    # Load the model
    device = get_device()
    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=8)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    evaluate(model, args.batch_size)
