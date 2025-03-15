import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os,sys
import numpy as np
import math

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,".."))

from dataset import coco_dataset


# 0. hyper param
class HyperParam:
    batch_size=64
    n_epoch=90
    lr_step_size=n_epoch//3
    learning_rate=0.001
    weight_decay=0.0001
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path=os.path.join(g_file_path,"yolo_v1.pth") # model path

    # label info
    S=7 # input image will be splitted into SxS anchors
    IMG_SIZE=224 # size of image
    GRID_SIZE=IMG_SIZE//S # size of grid
    BBOX_SIZE=int(4)
    CONFIDENT_SIZE=int(1) 
    # NUM_CLASS=int(90)
    NUM_CLASS=int(1)
    TARGET_CLASS_LABELS=[3] # 3 mean car
    OUT_DIM=int(NUM_CLASS+BBOX_SIZE+CONFIDENT_SIZE) # output dim, NUM_CLASS+BBOX_SIZE+CONFIDENT_SIZE

# 1. prepare dataset
class COCODataset(Dataset):
    def __init__(self, img_folder, anno_file_path, img_new_size=HyperParam.IMG_SIZE, transform=None):
        super().__init__()
        self.transform=transform
        self.img_new_size=img_new_size
        self.coco_parser=coco_dataset.COCOParser(img_dir=img_folder, annotation_file=anno_file_path)
        self.img_infos=self.coco_parser.get_img_infos()

    def __len__(self):
        return self.coco_parser.get_img_num()

    def __getitem__(self, index):
        """
        # label, [num_class,x,y,w,h,confidence]
        # image, [channel,width,height], range [0,255]
        """
        img_info=self.img_infos[index]
        img_id=self.coco_parser.get_img_id(img_info)

        # load and resize image
        img_pil=self.coco_parser.load_img(self.coco_parser.get_img_name(img_info=img_info))
        origin_width, origin_height=img_pil.size
        img_pil=coco_dataset.ImgLabelResize.image_resize(img=img_pil,new_size=self.img_new_size)
        new_width, new_height=img_pil.size

        # load and resize labels
        anno_infos=self.coco_parser.get_annotation_infos_by_img_id(img_id)
        labels=torch.zeros(HyperParam.S,HyperParam.S,HyperParam.OUT_DIM)

        for anno_info in anno_infos:
            # label, [num_class,x,y,w,h,confidence]
            # get annotation
            bbox=coco_dataset.ImgLabelResize.label_resize(origin_width,origin_height,anno_info['bbox'],self.img_new_size)
            
            # get category id
            category_id=anno_info['category_id']
            
            # only detect category in HyperParam.TARGET_CLASS_LABELS
            if category_id not in HyperParam.TARGET_CLASS_LABELS:
                continue
            
            # bounding box，[top_left_x,top_left_y,w,h]
            x,y,w,h=bbox[0],bbox[1],bbox[2],bbox[3]
            
            # grid coordinate
            x,y=x+w/2,y+h/2 # center of bbox
            grid_i,grid_j=int(x//HyperParam.GRID_SIZE),int(y//HyperParam.GRID_SIZE)

            # num class
            if grid_i>=HyperParam.S or grid_j>=HyperParam.S:
                print(f'anno_info["bbox"]:{bbox}, anno_info:{anno_info}')
                print(f'grid_i:{grid_i},grid_j:{grid_j},x:{x},y:{y},grid_size:{HyperParam.GRID_SIZE}')
                continue
            labels[grid_i,grid_j,0:HyperParam.NUM_CLASS]=torch.zeros(HyperParam.NUM_CLASS) # clear class
            labels[grid_i,grid_j,0]=1.0 # since there is only 1 class

            # normalize x,y,w,h
            x=(x-grid_i*HyperParam.GRID_SIZE)/HyperParam.GRID_SIZE
            y=(y-grid_j*HyperParam.GRID_SIZE)/HyperParam.GRID_SIZE
            w=w/new_width
            h=h/new_height
            labels[grid_i,grid_j,HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]=torch.tensor([x,y,w,h])
            # print(f'x,y,w,h{x,y,w,h}')

            # confidence
            confidence=1.0
            labels[grid_i,grid_j,HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]=confidence

        # convert pil image to torch tensor
        img_data:np.ndarray=np.array(img_pil)
        img_data:torch.Tensor=torch.from_numpy(img_data).float()
        img_data=img_data.permute(2,0,1) # channel, width, height
        img_data=img_data/255.0 # normalize

        if self.transform:
            img_data=self.transform(img_data)
        return img_data, labels

# 4.0 model define
class YOLO_V1(nn.Module):
    def __init__(self,input_channel=3, img_size=(HyperParam.IMG_SIZE,HyperParam.IMG_SIZE)):
        super().__init__()
        self.output_dim=HyperParam.OUT_DIM
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # (64,112,112)
        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # (192,56,56)
        # 第三层卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # (256,28,28)
        # 第四层卷积
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # (256,14,14)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256*14*14, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, HyperParam.S*HyperParam.S*HyperParam.OUT_DIM)
        )

    def forward(self,img):
        out=self.conv1(img)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        out=out.view(-1,256*14*14)
        out=self.fc(out)
        out=out.view(-1,HyperParam.S,HyperParam.S,HyperParam.OUT_DIM)

        pred_class=out[...,:HyperParam.NUM_CLASS]
        pred_coord=out[...,HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]
        pred_conf=out[...,HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]

        confidence_sigmoid=nn.Sigmoid()
        pred_conf=confidence_sigmoid(pred_conf)
        class_sigmoid=nn.Sigmoid()
        pred_class=class_sigmoid(pred_class)

        return pred_class, pred_coord, pred_conf

class YOLO_V1_Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self,pred_class:torch.Tensor, pred_bbox:torch.Tensor, pred_conf:torch.Tensor,labels:torch.Tensor):
        """
        predictions: (batch_size, S, S, num_class + x,y,w,h,confidence) -> YOLOv1 output with 1 bounding box
        target: (batch_size, S, S, num_class + x,y,w,h,confidence) -> Ground truth labels
        有目标和无目标的损失计算：
            有目标损失：
            loss_obj_coord:坐标损失乘以权重 5.0,再乘以 lambda_obj 掩码，只对有目标的网格单元计算损失。
            loss_obj_confidence:置信度损失乘以 lambda_obj 掩码。
            loss_obj_class:类别损失乘以 lambda_obj 掩码。
            无目标损失：
            loss_noobj_confidence:置信度损失乘以权重 0.5,再乘以 lambda_noobj 掩码，只对无目标的网格单元计算损失。
        """
        # Extract components
        # pred_class = predictions[..., :HyperParam.NUM_CLASS]    # Class probabilities
        # pred_conf = predictions[..., HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]   # Confidence score
        # pred_bbox = predictions[..., HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]  # Bounding box [x, y, w, h]

        target_class = labels[..., :HyperParam.NUM_CLASS]   # Ground truth class
        target_conf = labels[..., HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]  # Ground truth confidence
        target_bbox = labels[..., HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE] # Ground truth bounding box

        # mse loss function
        mse=nn.MSELoss(reduction='none')
        mae=nn.L1Loss(reduction='none')

        # if has obj or not for all grids
        lambda_obj=(target_conf>=0.9).float()
        lambda_noobj=(target_conf<0.9).float()

        # coordinate loss
        # print(f'pred_bbox.shape:{pred_bbox.shape}, target_bbox.shape:{target_bbox.shape}')
        # loss_coord=mse(pred_bbox[...,:2],target_bbox[...,:2]).sum(-1)+mse(
        #     torch.sqrt(torch.abs(pred_bbox[..., 2:4])+1e-8),
        #     torch.sqrt(torch.abs(target_bbox[..., 2:4])+1e-8)).sum(-1)
        loss_coord=mse(pred_bbox[...,:2],target_bbox[...,:2]).sum(-1)+\
            mse(torch.sqrt(pred_bbox[..., 2:4]+1e-12), torch.sqrt(target_bbox[..., 2:4]+1e-12)).sum(-1)
        # loss_coord=mse(pred_bbox[...,:2],target_bbox[...,:2]).sum(-1)+\
        #     mae(pred_bbox[..., 2:4], target_bbox[..., 2:4]).sum(-1)
        # print(f'pred_conf.shape:{pred_conf.shape}, target_conf.shape:{target_conf.shape}')
        loss_confidence=mse(pred_conf,target_conf)
        # print(f'lambda_obj.shape:{lambda_obj.shape}, loss_coord.shape:{loss_coord.shape}, loss_confidence.shape:{loss_confidence.shape}')

        # class loss
        # print(f'pred_class.shape:{pred_class.shape}, target_class.shape:{target_class.shape}')
        loss_class=mse(pred_class,target_class).sum(-1)

        # 有目标的损失
        loss_obj_coord = 5.0 * lambda_obj * loss_coord
        loss_obj_confidence = lambda_obj * loss_confidence
        loss_obj_class = lambda_obj * loss_class

        # 无目标的损失
        loss_noobj_confidence = 0.5 * lambda_noobj * loss_confidence

        # 总损失
        loss = loss_obj_coord.sum() + loss_obj_confidence.sum() + loss_obj_class.sum() + loss_noobj_confidence.sum()

        return loss