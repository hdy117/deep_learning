import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import os,sys
import matplotlib.pyplot as plt
import numpy as np

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,".."))

from dataset import coco_dataset
from dataset import cifar10_dataset

import resnet_base
import resnet18

# test dataloader
val_dataset=resnet_base.COCODataset(coco_dataset.coco_val_img_dir,
                        coco_dataset.coco_val_sub_annotation_file,
                        img_new_size=resnet_base.img_new_size,
                        target_class=resnet_base.target_class,
                        transform=resnet_base.transform)
# combine cifar-10 subset and coco subset
# combined_dataset=ConcatDataset([val_dataset, cifar10_dataset.test_dataset])
combined_dataset=ConcatDataset([val_dataset])
val_data_loader=DataLoader(dataset=combined_dataset, shuffle=True, 
                           batch_size=resnet_base.batch_size)

# test
def test():
    # load model
    model=resnet18.ResNet18(input_channel=3, out_dim=max(resnet_base.target_class))
    model=model.to(resnet_base.device)
    if os.path.exists(resnet_base.model_path):
        try:
            model.load_state_dict(torch.load(resnet_base.model_path))
            model.eval()
            print(f'Model loaded from {resnet_base.model_path}')
        except Exception as e:
            print(f'Error loading model: {e}')

    # Evaluation metrics
    total_samples = 0
    total_correct = 0
    total_true_positives = np.zeros(max(resnet_base.target_class))
    total_predicted_positives = np.zeros(max(resnet_base.target_class))
    total_actual_positives = np.zeros(max(resnet_base.target_class))
    conf_thresh = 0.51  # Adjust threshold if needed

    with torch.no_grad():
        print('================ Test ==================')
        for batch_idx, (samples, labels) in enumerate(val_data_loader):
            # Move data to device
            samples = samples.to(resnet_base.device)
            labels = labels.to(resnet_base.device)

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

# main
if __name__=="__main__":
    test()
