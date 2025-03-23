import torch
import os, sys
import numpy as np

from yolo_v1_transfer_learning import *

g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,"..","dataset"))

from dataset import bbox_utils

# test dataloader
val_dataset=COCODataset(coco_dataset.coco_val_img_dir,
                        coco_dataset.coco_val_sub_annotation_file,
                        img_new_size=HyperParam.IMG_SIZE,
                        transform=HyperParam.transform)
val_data_loader=DataLoader(dataset=val_dataset, 
                           shuffle=False, 
                           batch_size=HyperParam.batch_size)

# test
yolo_v1=YOLO_V1_Transfer()
yolo_v1=yolo_v1.to(HyperParam.device)

# draw and save bbox image
bbox_utils=bbox_utils.BBOXUtils(out_folder=os.path.join(g_file_path,"transfer_bbox_test"))

# update transfer learning model path
HyperParam.model_path=os.path.join(g_file_path, 'yolo_v1_transfer.pth')

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
        conf_thresh=0.5
        for batch_idx,(samples, labels) in enumerate(val_data_loader):
            # data to device
            samples=samples.to(HyperParam.device)
            labels=labels.to(HyperParam.device)

            # predict
            pred_class, pred_coord, pred_conf=yolo_v1.forward(samples)

            # save batch images with bbox
            bbox_utils.save_norm_imgs_with_bbox(samples, pred_class, pred_conf, 
                                          pred_coord, HyperParam.GRID_SIZE, conf_thresh)

            # loss
            batch_size=samples.shape[0]
            n_total+=batch_size*HyperParam.S*HyperParam.S
            label_conf=labels[...,HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]
            # print(f'{pred_class.shape},{label_class.shape}')

            # for only one class
            pred_obj=pred_conf>conf_thresh
            label_obj=label_conf>conf_thresh
            n_correct+=(pred_obj==label_obj).sum().item()

            # for multiple class
            # _,pred_idx=torch.max(pred_class,dim=1)
            # _,label_idx=torch.max(label_class,dim=1)
            # n_correct+=(pred_idx==label_idx).sum().item()
            print(f'batch idx:{batch_idx},accuracy:{n_correct/n_total}, n_correct:{n_correct}, n_total:{n_total}')

# main
if __name__=="__main__":
    test()
