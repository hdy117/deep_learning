import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import os,sys,time
import matplotlib.pyplot as plt

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,".."))

from dataset import coco_dataset
# from dataset import cifar10_dataset

import resnet18
import resnet_base

# hyper param
resnet_base.target_class=coco_dataset.coco_10_categories # coco_9_in_10_catetgories 
resnet_base.lr=1e-5
resnet_base.batch_size=128
resnet_base.n_epoch=20
resnet_base.lr_step_size=resnet_base.n_epoch//1
accumulate_steps=5

# train dataloader
train_dataset=resnet_base.COCODataset(coco_dataset.coco_train_img_dir,
                          coco_dataset.coco_train_annotation_file,
                          img_new_size=resnet_base.img_new_size,
                          target_class=resnet_base.target_class,
                          transform=resnet_base.transform)
# combine cifar-10 subset and coco subset
# combined_dataset=ConcatDataset([train_dataset, cifar10_dataset.train_dataset])
combined_dataset=ConcatDataset([train_dataset])
train_data_loader=DataLoader(dataset=combined_dataset, shuffle=True, 
                             batch_size=resnet_base.batch_size)

# train
residual_model=resnet18.ResNet18(input_channel=3, out_dim=resnet_base.out_dim)
residual_model=residual_model.to(resnet_base.device)

optimizer=torch.optim.Adam(residual_model.parameters(),lr=resnet_base.lr,
    weight_decay=resnet_base.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
    step_size=resnet_base.lr_step_size, gamma=0.1)
criterion=resnet18.ResidualLoss()

def train():
    # load saved model
    if os.path.exists(resnet_base.model_path):
        residual_model.load_state_dict(torch.load(resnet_base.model_path))
        residual_model.train()
        print(f'yolo v1 trained model loaded from {resnet_base.model_path}')

    # training
    for epoch in range(resnet_base.n_epoch):
        print('================train==================')
        t_start=time.time()
        for batch_idx,(samples, labels) in enumerate(train_data_loader):
            # data to resnet_base.device
            samples=samples.to(resnet_base.device)
            labels=labels.to(resnet_base.device)

            # predict
            y_pred=residual_model.forward(samples)

            # loss
            loss=criterion.forward(y_pred, labels)
            loss=loss/accumulate_steps

            # gradient 
            loss.backward()
            
            # gradient descent
            if (batch_idx+1)%accumulate_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            # loss
            if batch_idx%10==0:
                print(f'epoch:{epoch}, batch idx:{batch_idx},loss:{loss.item()}')

        # update learning rate
        scheduler.step()

        # save model
        torch.save(residual_model.state_dict(),resnet_base.model_path)

        # end of time
        t_end=time.time()
        print(f'elapsed time for one epoch is {t_end-t_start}')

# main
if __name__=="__main__":
    train()
