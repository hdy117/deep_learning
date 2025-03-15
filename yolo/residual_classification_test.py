import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os,sys
import matplotlib.pyplot as plt
import numpy as np

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,".."))

from residual_classification import *

# test dataloader
val_dataset=COCODataset(coco_dataset.coco_val_img_dir,
                        coco_dataset.coco_val_annotation_file,
                        img_new_size=img_new_size,
                        target_class=target_class,
                        transform=transform)
val_data_loader=DataLoader(dataset=val_dataset, shuffle=True, 
                           batch_size=batch_size)

# test
def test():
    # load model
    model=ResidualClassification(input_channel=3, out_dim=max(target_class))
    model=model.to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f'Model loaded from {model_path}')
        except Exception as e:
            print(f'Error loading model: {e}')

    # Evaluation metrics
    total_samples = 0
    total_correct = 0
    total_precision = 0
    total_recall = 0
    conf_thresh = 0.6  # Adjust threshold if needed

    with torch.no_grad():
        print('================ Test ==================')
        for batch_idx, (samples, labels) in enumerate(val_data_loader):
            # Move data to device
            samples = samples.to(device)
            labels = labels.to(device)

            # Predict
            y_pred = model(samples).sigmoid()  # Apply sigmoid activation

            # Convert predictions to binary (0 or 1)
            y_pred_bin = (y_pred > conf_thresh).float()

            # Calculate batch metrics
            batch_correct = (y_pred_bin == labels).sum().item()
            batch_total = labels.numel()

            # Precision and recall calculations
            true_positives = (y_pred_bin * labels).sum(dim=0).cpu().numpy()
            predicted_positives = y_pred_bin.sum(dim=0).cpu().numpy()
            actual_positives = labels.sum(dim=0).cpu().numpy()

            # Avoid division by zero
            precision = np.divide(true_positives, predicted_positives, out=np.zeros_like(true_positives), where=predicted_positives > 0)
            recall = np.divide(true_positives, actual_positives, out=np.zeros_like(true_positives), where=actual_positives > 0)

            # Aggregate results
            total_correct += batch_correct
            total_samples += batch_total
            total_precision += precision.mean()
            total_recall += recall.mean()

            # Print sample predictions
            if batch_idx % 10 == 0:
                for batch in range(min(3, samples.shape[0])):  # Show up to 3 samples
                    label_list = [f'{int(val)}' for val in labels[batch].tolist()]
                    pred_list = [f'{int(val)}' for val in y_pred_bin[batch].tolist()]
                    print(f'Label: {label_list}, Pred: {pred_list}')

        # Final Metrics
        accuracy = total_correct / total_samples
        avg_precision = total_precision / len(val_data_loader)
        avg_recall = total_recall / len(val_data_loader)

        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Test Precision: {avg_precision:.4f}')
        print(f'Test Recall: {avg_recall:.4f}')

# main
if __name__=="__main__":
    test()
