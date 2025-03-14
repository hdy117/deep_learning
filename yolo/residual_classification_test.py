import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os,sys
import matplotlib.pyplot as plt

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,".."))

from residual_classification import *

# test dataloader
val_dataset=COCODataset(coco_dataset.coco_val_img_dir,
                        coco_dataset.coco_val_sub_annotation_file,
                        img_new_size=img_new_size,
                        target_class=target_class,
                        transform=transform)
val_data_loader=DataLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size)

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

    # testing
    with torch.no_grad():
        print('================test==================')
        n_total=0
        n_correct=0
        conf_thresh=0.9
        for batch_idx,(samples, labels) in enumerate(val_data_loader):
            # data to device
            samples=samples.to(device)
            labels=labels.to(device)

            # predict
            y_pred=model.forward(samples)

            # size
            batch_size=y_pred.shape[0]
            out_dim=y_pred.shape[1]

            # calculate correction
            y_pred_obj=(y_pred>conf_thresh).float()
            correct_per_sample=(y_pred_obj==labels).sum().item()

            # accuracy
            n_correct += correct_per_sample
            n_total += out_dim * batch_size

        # for multiple class
        print(f'test accuracy:{n_correct/n_total}, n_correct:{n_correct}, n_total:{n_total}')

# main
if __name__=="__main__":
    test()
