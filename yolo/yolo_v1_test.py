import torch
import os
import numpy as np

from yolo_v1 import *

g_file_path=os.path.dirname(os.path.abspath(__file__))

# test dataloader
val_dataset=COCODataset(coco_dataset.coco_val_img_dir,coco_dataset.coco_val_annotation_file)
val_data_loader=DataLoader(dataset=val_dataset, shuffle=True, batch_size=HyperParam.batch_size)

# test
yolo_v1=YOLO_V1()
yolo_v1=yolo_v1.to(HyperParam.device)

def test():
    # load saved model
    if os.path.exists(HyperParam.model_path):
        yolo_v1.load_state_dict(torch.load(HyperParam.model_path))
        yolo_v1.eval()
        print(f'yolo v1 model loaded from {HyperParam.model_path}')
    else:
        print(f'error, no yolo v1 model found, {HyperParam.model_path}')
        return

    # testing
    with torch.no_grad():
        print('================test==================')
        n_total=0
        n_correct=0
        for batch_idx,(samples, labels) in enumerate(val_data_loader):
            # data to device
            samples=samples.to(HyperParam.device)
            labels=labels.to(HyperParam.device)

            # predict
            y_pred=yolo_v1.forward(samples)

            # loss
            pred_class=y_pred[...,HyperParam.NUM_CLASS]
            label_class=labels[...,HyperParam.NUM_CLASS]
            _,pred_idx=torch.max(pred_class,dim=1)
            _,label_idx=torch.max(label_class,dim=1)

            n_total+=samples.shape[0]
            n_correct+=(pred_idx==label_idx).sum().item()
            print(f'batch idx:{batch_idx},accuracy:{n_correct/n_total}, n_correct:{n_correct}, n_total:{n_total}')

# main
if __name__=="__main__":
    test()
