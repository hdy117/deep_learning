
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
import os,sys
import matplotlib.pyplot as plt

# global file path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,'..'))

from dataset import cifar10_dataset
from cifar10_vit import *

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset and dataloader
test_dataset = cifar10_dataset.CustomCIFAR10Dataset(data_dir=cifar10_dataset.data_dir, train=False, transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# test
def test():
    # load model
    model=CIFAR10_ViT(img_channel=3, img_size=[img_size,img_size],patch_size=patch_size,num_classes=num_classes)
    model=model.to(device)
    if os.path.exists(torch_model_path):
        try:
            model.load_state_dict(torch.load(torch_model_path))
            model.eval()
            print(f'Model loaded from {torch_model_path}')
        except Exception as e:
            print(f'Error loading model: {e}')

    # Evaluation metrics
    total_samples = 0
    total_correct = 0
    total_true_positives = np.zeros(num_classes)
    total_predicted_positives =np.zeros(num_classes)
    total_actual_positives =np.zeros(num_classes)
    conf_thresh = 0.6  # Adjust threshold if needed

    with torch.no_grad():
        print('================ Test ==================')
        for batch_idx, (samples, labels) in enumerate(test_loader):
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

            # Aggregate results
            total_correct += batch_correct
            total_samples += batch_total
            total_true_positives += true_positives
            total_predicted_positives += predicted_positives
            total_actual_positives += actual_positives

            # Print sample predictions
            if batch_idx % 10 == 0:
                for batch in range(min(3, samples.shape[0])):  # Show up to 3 samples
                    label_list = [f'{int(val)}' for val in labels[batch].tolist()]
                    pred_list = [f'{int(val)}' for val in y_pred_bin[batch].tolist()]
                    print(f'Label: {label_list}, Pred: {pred_list}')

        # Final Metrics
        accuracy = total_correct / total_samples
        precision = np.divide(total_true_positives, total_predicted_positives,
                              out=np.zeros_like(total_true_positives), where=total_predicted_positives > 0)
        recall = np.divide(total_true_positives, total_actual_positives,
                           out=np.zeros_like(total_true_positives), where=total_actual_positives > 0)
        avg_precision = precision.mean()
        avg_recall = recall.mean()

        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Test Precision: {avg_precision:.4f}, precision:{np.array2string(precision, precision=3, suppress_small=True)}')
        print(f'Test Recall: {avg_recall:.4f}, recall:{np.array2string(recall, precision=3, suppress_small=True)}')


if __name__ =="__main__":
    test()
