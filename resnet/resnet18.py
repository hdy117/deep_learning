import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os,sys,time

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,".."))

# residual block
class ResConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

        # a conv layer
        self.conv_layer=nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=kernel_size,padding=kernel_size//2),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=self.out_channels,out_channels=self.out_channels,kernel_size=kernel_size,padding=kernel_size//2),
        )

        # shortcut
        self.shortcut=(nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,kernel_size=1) \
            if self.in_channels!=self.out_channels else nn.Identity())

    def forward(self, x):
        # conv layer
        conv_layer=self.conv_layer(x)
        # shortcut
        shortcut=self.shortcut(x)
        # residual
        out=conv_layer+shortcut

        return out

class ResidualFeatures(nn.Module):
    def __init__(self,input_channel=3):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=3//2), 
            nn.MaxPool2d(2,2),  # (32,112,112)
            ResConv2dBlock(in_channels=32,out_channels=64,kernel_size=3),      
            ResConv2dBlock(in_channels=64,out_channels=64,kernel_size=3),      
            nn.MaxPool2d(2,2),  # (64,56,56)
            ResConv2dBlock(in_channels=64,out_channels=128,kernel_size=3),     
            ResConv2dBlock(in_channels=128,out_channels=128,kernel_size=3),     
            nn.MaxPool2d(2,2),  # (128,28,28)
            ResConv2dBlock(in_channels=128,out_channels=256,kernel_size=3),    
            ResConv2dBlock(in_channels=256,out_channels=256,kernel_size=3),    
            nn.MaxPool2d(2,2),  # (256,14,14)
            ResConv2dBlock(in_channels=256,out_channels=512,kernel_size=3),   
            ResConv2dBlock(in_channels=512,out_channels=512,kernel_size=3),   
            nn.MaxPool2d(2,2),  # (512,7,7)
        )
    
    def forward(self,x):
        '''
        residual conv2d feature, output is [batch_size, 1024, 7, 7]
        '''
        out=self.features(x)
        return out

# residual classification
class ResNet18(nn.Module):
    def __init__(self,input_channel=3, out_dim=1):
        super().__init__()
        self.output_dim=out_dim
        # feature extractor
        self.features=ResidualFeatures(input_channel=input_channel)

        # fc
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512*7*7),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512*7*7, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, self.output_dim),
        )

    def forward(self,img):
        out=self.features(img)
        out=out.view(-1,512*7*7)
        out=self.fc(out)
        return out

# residual loss
class ResidualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.BCEWithLogitsLoss(reduction='sum')
    
    def forward(self,y_pred,label):
        loss=self.loss(y_pred,label)
        return loss

