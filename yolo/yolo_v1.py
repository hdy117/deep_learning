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
    n_epoch=30
    learning_rate=0.01
    weight_decay=0.0005
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path=os.path.join(g_file_path,"yolo_v1.pth") # model path

    # label info
    S=7 # input image will be splitted into SxS anchors
    IMG_SIZE=224 # size of image
    GRID_SIZE=IMG_SIZE//S # size of grid
    BBOX_SIZE=int(4)
    CONFIDENT_SIZE=int(1) 
    # NUM_CLASS=int(90)
    NUM_CLASS=int(3)
    OUT_DIM=int(BBOX_SIZE+CONFIDENT_SIZE+NUM_CLASS) # output dim

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
        img_name=self.coco_parser.get_img_name(img_info=img_info)
        img_pil=self.coco_parser.load_img(self.coco_parser.get_img_name(img_info=img_info))
        width, height=img_pil.size
        img_pil=coco_dataset.ImgLabelResize.image_resize(img_pil,new_size=self.img_new_size)

        # load and resize labels
        anno_infos=self.coco_parser.get_annotation_infos_by_img_id(img_id)
        labels=torch.zeros(HyperParam.S,HyperParam.S,HyperParam.OUT_DIM)
        for i in range(HyperParam.S):
            for j in range(HyperParam.S):
                labels[i,j]=torch.zeros(HyperParam.OUT_DIM)

        for anno_info in anno_infos:
            # label, [num_class,x,y,w,h,confidence]
            # get annotation
            anno_info['bbox']=coco_dataset.ImgLabelResize.label_resize(width,height,anno_info['bbox'],self.img_new_size)
            
            # get category id
            category_id=anno_info['category_id']
            
            # only detect category less than or equal to HyperParam.NUM_CLASS
            if category_id>HyperParam.NUM_CLASS:
                continue
            
            # bounding box
            x,y,w,h=anno_info['bbox'][0],anno_info['bbox'][1],anno_info['bbox'][2],anno_info['bbox'][3]
            
            # grid coordinate
            i,j=int(x//HyperParam.GRID_SIZE),int(y//HyperParam.GRID_SIZE)

            # num class
            if i>=HyperParam.S or j>=HyperParam.S:
                print(f'anno_info["bbox"]:{anno_info['bbox']}')
                print(f'i:{i},j:{j},x:{x},y:{y},grid_size:{HyperParam.GRID_SIZE}')
            labels[i,j,0:HyperParam.NUM_CLASS]=torch.zeros(HyperParam.NUM_CLASS) # clear class
            labels[i,j,category_id-1]=1.0

            # normalize x,y,w,h
            x=(x-i*HyperParam.GRID_SIZE)/HyperParam.GRID_SIZE
            y=(y-i*HyperParam.GRID_SIZE)/HyperParam.GRID_SIZE
            w=w/width
            h=h/height
            labels[i,j,HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]=torch.tensor([x,y,w,h])

            # confidence
            confidence=1.0
            labels[i,j,HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]=confidence

        # convert pil image to torch tensor
        img_data:np.ndarray=np.array(img_pil)
        img_data:torch.Tensor=torch.from_numpy(img_data)
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
        self.conv_layers=nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=32,kernel_size=7,padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2,stride=2),  # (32,112,112)

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2,stride=2),  # (64,56,56)

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2,stride=2),  # (256,28,28)

            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2,stride=2)  # (512,14,14)
        )

        self.fc=nn.Sequential(
            nn.Linear(in_features=512*14*14,out_features=2048),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=2048,out_features=2048),
            nn.Tanh(),
            nn.Linear(in_features=2048,out_features=7*7*self.output_dim)
        )

    def forward(self,img):
        out=self.conv_layers(img)
        out=out.view(-1,512*14*14)
        out=self.fc(out)
        out=out.view(-1,HyperParam.S,HyperParam.S,HyperParam.OUT_DIM)
        return out

class YOLO_V1_Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self,predictions:torch.Tensor,labels:torch.Tensor):
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
        pred_class = predictions[..., :HyperParam.NUM_CLASS]    # Class probabilities
        pred_conf = predictions[..., HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]   # Confidence score
        pred_bbox = predictions[..., HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]  # Bounding box [x, y, w, h]

        target_class = labels[..., :HyperParam.NUM_CLASS]   # Ground truth class
        target_conf = labels[..., HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]  # Ground truth confidence
        target_bbox = labels[..., HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE] # Ground truth bounding box

        # mse loss function
        mse=nn.MSELoss(reduce='none')

        # if has obj or not
        lambda_obj=target_conf>=0.5
        lambda_noobj=target_conf<0.5

        # coordinate loss
        loss_coord=mse(pred_bbox[...,:2],target_bbox[...,:2])+mse(
            torch.sqrt(torch.abs(pred_bbox[..., 2:4]) + 1e-8),
            torch.sqrt(torch.abs(target_bbox[..., 2:4]) + 1e-8)
        )
        loss_confidence=mse(pred_conf,target_conf)

        # class loss
        loss_class=mse(pred_class,target_class)

        # 有目标的损失
        loss_obj_coord = 5.0 * lambda_obj * loss_coord
        loss_obj_confidence = lambda_obj * loss_confidence
        loss_obj_class = lambda_obj * loss_class

        # 无目标的损失
        loss_noobj_confidence = 0.5 * lambda_noobj * loss_confidence

        # 总损失
        loss = loss_obj_coord.sum() + loss_obj_confidence.sum() + loss_obj_class.sum() + loss_noobj_confidence.sum()

        return loss