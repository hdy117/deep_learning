import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os,sys
import matplotlib.pyplot as plt

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,".."))

from dataset import coco_dataset

# coco dataset
class COCODataset(Dataset):
    def __init__(self, img_folder, anno_file_path, img_new_size=224, target_class:list=[1,2,3], transform=None):
        super().__init__()
        self.transform=transform
        self.target_class=target_class
        self.img_new_size=img_new_size
        self.coco_parser=coco_dataset.COCOParser(img_dir=img_folder, annotation_file=anno_file_path)
        self.img_infos=self.coco_parser.get_img_infos()

    def __len__(self):
        return self.coco_parser.get_img_num()

    def __getitem__(self, index):
        """
        # label, [num_class]
        # image, [channel,width,height], range [0,255]
        """
        img_info=self.img_infos[index]
        img_id=self.coco_parser.get_img_id(img_info)

        # load and resize image
        img_pil=self.coco_parser.load_img(self.coco_parser.get_img_name(img_info=img_info))
        # origin_width, origin_height=img_pil.size
        img_pil=coco_dataset.ImgLabelResize.image_resize(img=img_pil,new_size=self.img_new_size)
        # new_width, new_height=img_pil.size

        # labels
        labels=torch.zeros(max(self.target_class))

        # load and resize labels
        anno_infos=self.coco_parser.get_annotation_infos_by_img_id(img_id)

        for anno_info in anno_infos:
            # get category id
            category_id=anno_info['category_id']
            
            # only detect category in HyperParam.TARGET_CLASS_LABELS
            if category_id in self.target_class:
                # num class
                labels[category_id-1]=1.0 # class

        # show img
        # print(f'labels:{labels}')
        # img_pil.show()

        if self.transform:
            img_pil=self.transform(img_pil)
        
        # print img
        # print(f'img_pil:{img_pil.mean()}')

        return img_pil, labels

# residual block
class ResConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

        # a bottle neck 
        self.bottle_neck=nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels//2,kernel_size=1),
            nn.BatchNorm2d(self.out_channels//2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.out_channels//2,out_channels=self.out_channels//2,kernel_size=kernel_size,padding=kernel_size//2),
            nn.BatchNorm2d(self.out_channels//2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.out_channels//2,out_channels=self.out_channels-self.in_channels,kernel_size=1)
        )

        # self.residual=nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,kernel_size=1)

    def forward(self, x):
        # residual bottle neck
        bottle_neck=self.bottle_neck(x)
        # residual=self.residual(x)
        # out=torch.concat([bottle_neck,residual],dim=1)
        out=torch.concat([bottle_neck,x],dim=1)

        # avg pool 2d
        avg_pool2d=nn.AvgPool2d(2,2)
        out=avg_pool2d(out)

        return out

# residual classification
class ResidualClassification(nn.Module):
    def __init__(self,input_channel=3, out_dim=1):
        super().__init__()
        self.output_dim=out_dim
        # conv0
        self.conv0=nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=8,kernel_size=3,padding=1),
        )
        # conv1
        self.conv1=ResConv2dBlock(in_channels=8,out_channels=32) # (32,112,112)
        # conv2
        self.conv2=ResConv2dBlock(in_channels=32,out_channels=64) # (64,56,56)
        # conv3
        self.conv3=ResConv2dBlock(in_channels=64,out_channels=128) # (128,28,28)
        # conv4
        self.conv4=ResConv2dBlock(in_channels=128,out_channels=256) # (256,14,14)
        # fc
        self.fc = nn.Sequential(
            nn.Linear(256*14*14, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096, self.output_dim),
            nn.Sigmoid()
        )

    def features(self,x):
        '''
        residual conv2d feature, output is [batch_size, 256, 14, 14]
        '''
        out=self.conv0(x)
        out=self.conv1(out)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        return out

    def forward(self,img):
        out=self.features(img)
        out=out.view(-1,256*14*14)
        out=self.fc(out)
        return out

# residual loss
class ResidualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.BCELoss(reduction='sum')
    
    def forward(self,y_pred,label):
        loss=self.loss(y_pred,label)
        return loss

# hyper param
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path=os.path.join(g_file_path,"residual_classification.pth")
batch_size=64
n_epoch=6
img_new_size=224
target_class=[1,2,3,4,5,6,7,8,9,10] # coco category [person, bicycle, car]

# train dataloader
transform = transforms.Compose([
    # 将 PIL 图像转换为 PyTorch 张量
    transforms.ToTensor(),
    # 对图像进行归一化处理
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# train dataloader
train_dataset=COCODataset(coco_dataset.coco_train_img_dir,
                          coco_dataset.coco_train_sub_annotation_file,
                          img_new_size=img_new_size,
                          target_class=target_class,
                          transform=transform)
train_data_loader=DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

# train
residual_model=ResidualClassification(input_channel=3, out_dim=max(target_class))
residual_model=residual_model.to(device)

optimizer=torch.optim.Adam(residual_model.parameters(),lr=0.001,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
criterion=ResidualLoss()

def train():
    # load saved model
    if os.path.exists(model_path):
        residual_model.load_state_dict(torch.load(model_path))
        residual_model.train()
        print(f'yolo v1 trained model loaded from {model_path}')

    # training
    for epoch in range(n_epoch):
        print('================train==================')
        for batch_idx,(samples, labels) in enumerate(train_data_loader):
            # data to device
            samples=samples.to(device)
            labels=labels.to(device)

            # clear gradient
            optimizer.zero_grad()

            # predict
            y_pred=residual_model.forward(samples)

            # loss
            loss=criterion.forward(y_pred, labels)

            # gradient descent
            loss.backward()
            optimizer.step()

            # loss
            if batch_idx%10==0:
                print(f'epoch:{epoch}, batch idx:{batch_idx},loss:{loss.item()}')

        # update learning rate
        scheduler.step()

        # save model
        torch.save(residual_model.state_dict(),model_path)

        # test 
        test()

# main
if __name__=="__main__":
    train()
