import torch
from torch.utils.data import DataLoader
import os

from yolo_v1 import *

# train dataloader
train_dataset=COCODataset(coco_dataset.coco_train_img_dir,coco_dataset.coco_train_sub_annotation_file)
train_data_loader=DataLoader(dataset=train_dataset, shuffle=True, batch_size=HyperParam.batch_size)

# train
yolo_v1=YOLO_V1()
yolo_v1=yolo_v1.to(HyperParam.device)

optimizer=torch.optim.Adam(yolo_v1.parameters(),lr=HyperParam.learning_rate,weight_decay=HyperParam.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion=YOLO_V1_Loss()

def train():
    # load saved model
    if os.path.exists(HyperParam.model_path):
        yolo_v1.load_state_dict(torch.load(HyperParam.model_path))
        yolo_v1.eval()
        print(f'yolo v1 trained model loaded from {HyperParam.model_path}')

    # training
    for epoch in range(HyperParam.n_epoch):
        print('================train==================')
        for batch_idx,(samples, labels) in enumerate(train_data_loader):
            # data to device
            samples=samples.to(HyperParam.device)
            labels=labels.to(HyperParam.device)

            # clear gradient
            optimizer.zero_grad()

            # predict
            pred_class, pred_coord, pre_confidence=yolo_v1.forward(samples)

            # loss
            loss=criterion(pred_class, pred_coord, pre_confidence, labels)

            # gradient descent
            loss.backward()
            optimizer.step()

            # loss
            if batch_idx%10==0:
                print(f'epoch:{epoch}, batch idx:{batch_idx},loss:{loss.item()}')

        # update learning rate
        scheduler.step()

        # save model
        torch.save(yolo_v1.state_dict(),HyperParam.model_path)

# main
if __name__=="__main__":
    train()
