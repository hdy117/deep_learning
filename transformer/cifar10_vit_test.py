
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

# transform for dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size,img_size),interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
])

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
    total_true_positives = torch.zeros(num_classes)
    total_predicted_positives =torch.zeros(num_classes)
    total_actual_positives =torch.zeros(num_classes)
    
    softmax=nn.Softmax(dim=1)

    with torch.no_grad():
        print('================ Test ==================')
        conf_thresh=0.0
        for batch_idx, (samples, labels) in enumerate(test_loader):
            # Move data to device
            samples = samples.to(device)
            labels = labels.to(device)
            
            N=samples.shape[0] # batch size

            # Predict
            y_pred = model(samples) 
            y_pred = softmax(y_pred)

            # Convert predictions to binary (0 or 1)
            y_pred_label=torch.argmax(y_pred, dim=1) # pred label
            y_pred_bin = torch.zeros(N, num_classes, device=y_pred.device)
            y_pred_bin.scatter_(dim=1, index=y_pred_label.unsqueeze(1), value=1.0)

            # Calculate batch metrics
            batch_correct = (y_pred_bin == labels).float().sum().item()
            batch_total = N

            # Precision and recall calculations, bigger than conf_thresh will make sure it is a true positive
            true_positives = ((y_pred_bin==labels)*(y_pred_bin>0.5)).float().sum(dim=0).cpu()
            predicted_positives = y_pred_bin.sum(dim=0).cpu()
            actual_positives = labels.sum(dim=0).cpu()

            # Aggregate results
            total_correct += batch_correct
            total_samples += batch_total*num_classes
            
            total_true_positives += true_positives
            total_predicted_positives += predicted_positives
            total_actual_positives += actual_positives

            # Print sample predictions
            if batch_idx % 10 == 0:
                print(f'***********************')
                for batch in range(min(1, samples.shape[0])):  # Show up to 3 samples
                    label_list = [f'{int(val)}' for val in labels[batch].tolist()]
                    pred_list = [f'{int(val)}' for val in y_pred_bin[batch].tolist()]
                    print(f'Label: {label_list}, Pred: {pred_list}')

        # Final Metrics
        epslion=1e-8
        accuracy = total_correct / total_samples
        precision = torch.divide(total_true_positives, total_predicted_positives+epslion).numpy()
        recall = torch.divide(total_true_positives, total_actual_positives+epslion).numpy()
        avg_precision = precision.mean()
        avg_recall = recall.mean()

        print(f'total_true_positives:{total_true_positives}')
        print(f'total_predicted_positives:{total_predicted_positives}')
        print(f'total_actual_positives:{total_actual_positives}')
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Test Precision: {avg_precision:.4f}, precision:{np.array2string(precision, precision=3, suppress_small=True)}')
        print(f'Test Recall: {avg_recall:.4f}, recall:{np.array2string(recall, precision=3, suppress_small=True)}')


if __name__ =="__main__":
    test()
