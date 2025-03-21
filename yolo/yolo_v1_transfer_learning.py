import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os,sys,time
import numpy as np
import torchvision.models as models

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,".."))

from dataset import coco_dataset
from yolo_v1 import *
import residual_classification

# 4.0 model define
class YOLO_V1_Transfer(nn.Module):
    def __init__(self,input_channel=3, img_size=(HyperParam.IMG_SIZE,HyperParam.IMG_SIZE)):
        super().__init__()
        self.output_dim=HyperParam.OUT_DIM
        # self.vgg = models.vgg16(pretrained=True)
        # self.vgg_features = self.vgg.features
        self.residual=residual_classification.ResidualClassification(input_channel=3,out_dim=residual_classification.out_dim)
        self.residual.load_state_dict(torch.load(residual_classification.model_path))
        self.residual.eval()
        self.residual_feature=self.residual.features

        # 全连接层
        self.fc = nn.Sequential(
            nn.BatchNorm1d(1024*7*7),
            nn.LeakyReLU(),
            nn.Linear(1024*7*7, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, HyperParam.S*HyperParam.S*HyperParam.OUT_DIM),
            nn.Sigmoid()
        )

    def forward(self,img):
        out= x = self.residual_feature(img)
        out=out.view(-1,1024*7*7)
        out=self.fc(out)
        out=out.view(-1,HyperParam.S,HyperParam.S,HyperParam.OUT_DIM)

        pred_class=out[...,:HyperParam.NUM_CLASS]
        pred_coord=out[...,HyperParam.NUM_CLASS:HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]
        pred_conf=out[...,HyperParam.NUM_CLASS+HyperParam.BBOX_SIZE]

        return pred_class, pred_coord, pred_conf
    
# train dataloader
train_dataset=COCODataset(coco_dataset.coco_train_img_dir,
                          coco_dataset.coco_train_sub_annotation_file,
                          img_new_size=HyperParam.IMG_SIZE,
                          transform=HyperParam.transform)
train_data_loader=DataLoader(dataset=train_dataset, 
                             shuffle=True,
                             batch_size=HyperParam.batch_size)

# train
yolo_v1_transfer=YOLO_V1_Transfer()
yolo_v1_transfer=yolo_v1_transfer.to(HyperParam.device)

# update transfer learning model path
HyperParam.model_path=os.path.join(g_file_path, 'yolo_v1_transfer.pth')

# 冻结 VGG 模型的卷积层
for param in yolo_v1_transfer.residual.features.parameters():
    param.requires_grad = False

# optimizer
optimizer=torch.optim.Adam(yolo_v1_transfer.parameters(),lr=HyperParam.learning_rate,weight_decay=HyperParam.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=HyperParam.lr_step_size, gamma=0.1)
criterion=YOLO_V1_Loss()

def train():
    # load saved model
    if os.path.exists(HyperParam.model_path):
        yolo_v1_transfer.load_state_dict(torch.load(HyperParam.model_path))
        yolo_v1_transfer.train()
        print(f'yolo v1 trained model loaded from {HyperParam.model_path}')

    # training
    for epoch in range(HyperParam.n_epoch):
        print('================train==================')
        t_start=time.time()
        for batch_idx,(samples, labels) in enumerate(train_data_loader):
            # data to device
            samples=samples.to(HyperParam.device)
            labels=labels.to(HyperParam.device)

            # clear gradient
            optimizer.zero_grad()

            # predict
            pred_class, pred_coord, pre_confidence=yolo_v1_transfer.forward(samples)

            # loss
            loss=criterion(pred_class, pred_coord, pre_confidence, labels)

            # gradient descent
            loss.backward()
            optimizer.step()

            # loss
            if batch_idx%10==0:
                # if loss < best_loss:
                #     # save best model
                #     torch.save(yolo_v1_transfer.state_dict(),HyperParam.model_path)
                #     best_loss=loss
                #     print(f'********save best model with loss:{best_loss}***********')
                print(f'epoch:{epoch}, batch idx:{batch_idx},loss:{loss.item()}')

        # update learning rate
        scheduler.step()

        # save
        torch.save(yolo_v1_transfer.state_dict(),HyperParam.model_path)

        t_end=time.time()
        print(f'elapsed time for one batch {t_end-t_start}')

if __name__=="__main__":
    train()