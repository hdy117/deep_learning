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
    batch_size=512
    n_epoch=30
    learning_rate=0.01
    weight_decay=0.0005
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path=os.path.join(g_file_path,"yolo_v1.pth") # model path

    # label info
    S=7 # input image will be splitted into SxS anchors
    IMG_SIZE=224 # size of image
    GRID_SIZE=IMG_SIZE/S # size of grid
    BBOX_SIZE=4
    CONFIDENT_SIZE=1 
    NUM_CLASS=90
    OUT_DIM=BBOX_SIZE+CONFIDENT_SIZE+NUM_CLASS # output dim

# 1. prepare dataset
class COCODataset(Dataset):
    def __init__(self, img_folder, anno_file_path, img_new_size=224, transform=None):
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
        img_pil=coco_dataset.ImgLabelResize.image_resize(img_pil,new_size=self.img_new_size)
        width, height=img_pil.size

        # load and resize labels
        anno_infos=self.coco_parser.get_annotation_infos_by_img_id(img_id)
        labels=torch.zeros(HyperParam.S,HyperParam.S,HyperParam.OUT_DIM)
        for i in range(HyperParam.S):
            for j in range(HyperParam.S):
                labels[i,j]=torch.zeros(HyperParam.OUT_DIM)

        for anno_info in anno_infos:
            anno_info['bbox']=coco_dataset.ImgLabelResize.label_resize(width, height,anno_info['bbox'],self.img_new_size)
            category_id=anno_info['category_id']
            x,y,w,h=anno_info['bbox'][0],anno_info['bbox'][1],anno_info['bbox'][2],anno_info['bbox'][3]
            # label, [num_class,x,y,w,h,confidence]
            i,j=x//HyperParam.GRID_SIZE,y//HyperParam.GRID_SIZE

            # num class
            labels[i,j,0:HyperParam.NUM_CLASS]=torch.zeros(HyperParam.NUM_CLASS) # clear class
            labels[i,j,category_id-1]=1.0

            # normalize x,y,w,h
            x=(x-i*HyperParam.GRID_SIZE)/HyperParam.GRID_SIZE
            y=(y-i*HyperParam.GRID_SIZE)/HyperParam.GRID_SIZE
            w=w/width
            h=h/height
            labels[i,j,HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.OUT_DIM]=torch.tensor([x,y,w,h])

            # confidence
            confidence=1.0
            labels[i,j,HyperParam.NUM_CLASS+HyperParam.OUT_DIM]=confidence

        # convert pil image to torch tensor
        img_data:np.ndarray=np.array(img_pil)
        img_data:torch.Tensor=torch.from_numpy(img_data)
        img_data=img_data.permute(2,0,1) # channel, width, height
        img_data=img_data/255.0 # normalize

        if self.transform:
            img_data=self.transform(img_data)
        return img_data, anno_infos

# 4.0 model define
class YOLO_V1(nn.Module):
    def __init__(self,input_channel=3, img_size=(224,224)):
        super().__init__()
        self.output_dim=HyperParam.OUT_DIM
        self.conv_layers=nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=64,kernel_size=7,padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),  # (64,112,112)

            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),  # (192,56,56)

            nn.Conv2d(in_channels=192,out_channels=128,kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),  # (512,28,28)

            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),  # (1024,14,14)

            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),  # (1024,7,7)
        )

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024*7*7,out_features=4096),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096,out_features=4096),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096,out_features=7*7*self.output_dim),
        )

    def forward(self,img):
        out=self.conv_layers(img)
        out=self.fc(out)
        out=out(-1,HyperParam.S,HyperParam.S,HyperParam.OUT_DIM)
        return out

class YOLO_V1_Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self,predictions:torch.Tensor,labels:torch.Tensor):
        """
        predictions: (batch_size, S, S, num_class + x,y,w,h,confidence) -> YOLOv1 output with 1 bounding box
        target: (batch_size, S, S, num_class + x,y,w,h,confidence) -> Ground truth labels
        """
        # Extract components
        pred_class = predictions[..., :HyperParam.NUM_CLASS]    # Class probabilities
        pred_conf = predictions[..., HyperParam.NUM_CLASS+HyperParam.OUT_DIM]   # Confidence score
        pred_bbox = predictions[..., HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.OUT_DIM]  # Bounding box [x, y, w, h]

        target_class = labels[..., :HyperParam.NUM_CLASS]   # Ground truth class
        target_conf = labels[..., HyperParam.NUM_CLASS+HyperParam.OUT_DIM]  # Ground truth confidence
        target_bbox = labels[..., HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.OUT_DIM] # Ground truth bounding box

        # mse loss function
        mse=nn.MSELoss()

        # if has obj or not
        lambda_obj=float(target_conf>=0.5)
        lambda_noobj=float(target_conf<0.5)

        # coordinate loss
        loss_coord=mse(pred_bbox[...,:2],target_bbox[...,:2])+self.mse(
            torch.sqrt(torch.abs(pred_bbox[..., 2:4]) + 1e-8),
            torch.sqrt(torch.abs(target_bbox[..., 2:4]) + 1e-8)
        )
        loss_confidence=mse(pred_conf,target_conf)

        # total loss
        loss=lambda_obj*loss_coord*5+lambda_obj*loss_confidence+lambda_noobj*loss_confidence*0.5

        # class loss
        if lambda_obj>0.5:
            loss_class=mse(pred_class,target_class)
            loss=loss+loss_class
        
        return loss