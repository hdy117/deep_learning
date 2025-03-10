import torch
import os, sys
import numpy as np

from yolo_v1 import *

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,"..",'dataset'))

from dataset import coco_dataset
from dataset import bbox_utils

if __name__ == "__main__":
    coco_dataset=COCODataset(coco_dataset.coco_val_img_dir,anno_file_path=coco_dataset.coco_val_sub_annotation_file)
    bbox_saver=bbox_utils.BBOXUtils(out_folder='bbox_test')

    for idx,(sample,labels) in enumerate(coco_dataset):
        target_class = labels[..., :HyperParam.NUM_CLASS]   # Ground truth class
        target_conf = labels[..., HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]  # Ground truth confidence
        target_bbox = labels[..., HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE] # Ground truth bounding box
        bbox_saver.save_one_norm_img_with_bbox(sample, target_class, target_conf, target_bbox, grid_size=HyperParam.GRID_SIZE,conf_thresh=0.6,img_idx=idx)

