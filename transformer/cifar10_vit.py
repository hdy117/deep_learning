import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
import os,sys
import matplotlib.pyplot as plt

# global file path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,'..'))

from dataset import cifar10_dataset
import RoPE

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CIFAR10_ViT(nn.Module):
    def __init__(self,img_channel:int=3,img_size=[32,32], patch_size:int=16, num_classes=10):
        super().__init__()  
        self.img_w=img_size[0]
        self.img_h=img_size[1]
        self.img_channel=img_channel
        
        self.path_size=patch_size   # row/column of a patch, 8
        self.patch_num=(self.img_channel)*(self.img_w//self.path_size)*(self.img_h//self.path_size) # total patch number, 3*2*2-->12
        self.patch_pixel_num=self.path_size*self.path_size # pixel number in a patch, 16*16-->256
        self.num_classes=num_classes    # number of class, 10
        self.d_model=self.patch_pixel_num*4
        
        self.class_token = nn.Parameter(torch.randn(1, 1, self.d_model))  # 添加分类标记
        
        self.embedding = nn.Linear(self.patch_pixel_num, self.d_model)   # embedding
        self.RoPE=RoPE.RotaryPositionalEncoding(dim=self.d_model)
        self.transfomer_encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=16, batch_first=True,dim_feedforward=4*self.d_model),
            num_layers=24
        )

        # mapping feature to 10 at the end
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x:torch.Tensor):
        N, C, H, W = x.shape    # [512,3,32,32]

        # convert input to [batch, seq, feat]
        x=x.unfold(3,self.path_size,self.path_size).unfold(4,self.path_size,self.path_size)
        x=x.contiguous().view(N,self.patch_num,self.patch_pixel_num) # # [batch, seq, feat]
        
        # embedding and positional encoding
        x=self.embedding(x) # self.patch_pixel_num --> d_model
        
        class_tokens = self.class_token.expand(x.size(0), -1, -1) # add class token to capture global information
        x = torch.cat((class_tokens, x), dim=1)  # concat class_token and embedding
        
        x=self.RoPE(x)  # positional encoding
        
        # transformer encoder
        x=self.transfomer_encoder(x)

        # take class token to predict
        x = x[:,0,:].squeeze(1)

        # mapping to number of classes
        x = self.fc(x)

        return x

# hyper parameters
learning_rate=5e-4
n_epochs=30
lr_step_size=n_epochs//3
batch_size=128
img_size=32
num_classes=10
torch_model_path=os.path.join(g_file_path,".","ViT_cifar10.pth")
patch_size=16

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为torch.Tensor类型，同时将像素值归一化到[0, 1]
    # transforms.Normalize((0.1307,), (0.3081,))  # 对数据进行归一化，这里的均值和标准差是MNIST数据集的统计值
])

# 加载训练集
train_dataset = cifar10_dataset.CustomCIFAR10Dataset(data_dir=cifar10_dataset.data_dir, train=True, \
    transform=cifar10_dataset.transform_no_resize)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,  # 每个批次的样本数量
                                           shuffle=True)  # 是否在每个epoch打乱数据

# model
cifar10_vit=CIFAR10_ViT(img_channel=3, img_size=[img_size,img_size],patch_size=patch_size,num_classes=num_classes)
cifar10_vit=cifar10_vit.to(device)

# define train
def train():
    # criterion
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(cifar10_vit.parameters(), lr=learning_rate, weight_decay=5e-4)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size=lr_step_size, gamma=0.1)

    # load torch model
    if os.path.exists(torch_model_path):
        cifar10_vit.load_state_dict(torch.load(torch_model_path))
        cifar10_vit.train()

    # train
    for epoch in range(n_epochs):
        n_total=0
        n_correct=0
        print('====================================')
        for idx, (samples, labels) in enumerate(train_loader):
            # print(f"len(samples):{len(samples)}, {samples.shape}")
            # print(f"len(labels):{len(labels)}, {labels.shape}")
            # to device
            samples=samples.to(device)
            labels=labels.to(device)

            # prediction
            y_pred = cifar10_vit.forward(samples)

            # loss
            loss=criterion(y_pred,labels)

            # calculate gradients
            loss.backward()

            # gradient descent
            optimizer.step()
            optimizer.zero_grad()

            # accuracy
            _, predicted_indices = torch.max(y_pred, dim=1)
            _, label_indices = torch.max(labels, dim=1)
            n_total += y_pred.shape[0]
            n_correct += (predicted_indices == label_indices).sum().item()
            if idx%100==0:
                print(f'predicted_indices[0]:{predicted_indices[0]}, labels[0]:{label_indices[0]}')
                print(f'Epoch {epoch}, Step {idx}, accuracy:{n_correct / n_total}, n_total:{n_total}, n_correct:{n_correct}')
        
        # update learning rate
        scheduler.step()
        
        # save model
        torch.save(cifar10_vit.state_dict(), torch_model_path)

if __name__ =="__main__":
    # img, label=train_dataset[34]
    # img=img.permute(1,2,0)
    # print(f'img.shape:{img.shape}, img type:{type(img)}, label.shape:{label}, label.type:{type(label)}')
    # plt.imshow(img)
    # plt.title(f'{label}')
    # plt.show()
    
    # train
    train()
