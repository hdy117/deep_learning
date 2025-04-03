import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
import os,sys
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter

# global file path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,'..'))

from dataset import cifar10_dataset
import RoPE

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CIFAR10_ViT(nn.Module):
    def __init__(self,img_channel:int=3,img_size=[64,64], patch_size:int=8, num_classes=10):
        super().__init__()  
        self.img_w=img_size[0]
        self.img_h=img_size[1]
        self.img_channel=img_channel
        
        self.patch_size=patch_size   # row/column of a patch, 8
        self.patch_num=(self.img_w//self.patch_size)*(self.img_h//self.patch_size) # total patch number, 8*8-->64
        self.patch_pixel_num=self.img_channel*self.patch_size*self.patch_size # pixel number in a patch, 3*8*8-->192
        self.num_classes=num_classes    # number of class, 10
        self.d_model=max(576,self.patch_pixel_num) # make sure d_model is not less than 512
        
        self.class_token = nn.Parameter(torch.randn(1, 1, self.d_model))  # 添加分类标记
        
        self.embedding = nn.Linear(self.patch_pixel_num, self.d_model)   # embedding
        self.RoPE=RoPE.RotaryPositionalEncoding(dim=self.d_model)
        self.transfomer_encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=12, batch_first=True,activation='gelu',dim_feedforward=4*self.d_model),
            num_layers=12
        )

        # mapping feature to 10 at the end
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, self.num_classes),
        )

    def forward(self, x:torch.Tensor):
        N, C, H, W = x.shape    # [512,3,32,32]

        # convert input to [batch, seq, feat]
        x=x.unfold(2,self.patch_size,self.patch_size)
        x=x.unfold(3,self.patch_size,self.patch_size) # [batch,channel,num_patch_per_H,num_patch_per_W,patch_size,patch_size]
        x=x.permute(0, 2, 3, 1, 4, 5).contiguous().view(N,self.patch_num,self.patch_pixel_num) # [batch,num_patch_per_H,num_patch_per_W,channel,patch_size,patch_size]
        
        # embedding and positional encoding
        x=self.embedding(x) # self.patch_pixel_num --> d_model
        class_tokens = self.class_token.expand(x.size(0), -1, -1) # add class token to capture global information
        x = torch.cat((class_tokens, x), dim=1)  # concat class_token and embedding
        x=self.RoPE(x)  # positional encoding
        
        # transformer encoder
        x=self.transfomer_encoder(x)

        # take class token to predict
        x = x[:,0,:]

        # mapping to number of classes
        x = self.fc(x)

        return x

# hyper parameters
learning_rate=1e-6
n_epochs=40
lr_step_size=n_epochs//2
batch_size=350
img_size=64
num_classes=10
torch_model_path=os.path.join(g_file_path,".","ViT_cifar10.pth")
patch_size=8
accumulate_steps=2

# transform for dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size,img_size),interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
])

# train dataset
train_dataset = cifar10_dataset.CustomCIFAR10Dataset(data_dir=cifar10_dataset.data_dir, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True) 

# test dataset and dataloader
test_dataset = cifar10_dataset.CustomCIFAR10Dataset(data_dir=cifar10_dataset.data_dir, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

# model
cifar10_vit=CIFAR10_ViT(img_channel=3, img_size=[img_size,img_size],patch_size=patch_size,num_classes=num_classes)
cifar10_vit=cifar10_vit.to(device)

# define train
def train():
    # create tensorboard summary writter
    summary_writer=SummaryWriter(log_dir=os.path.join(g_file_path,'log','cifar10_vit_12_layer'))

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
        train_loss=0.0
        print('====================================')
        t_start=time.time()
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
            train_loss+=loss.item()
            loss=loss/accumulate_steps

            # calculate gradients
            loss.backward()

            # gradient descent
            if (idx+1)%accumulate_steps==0:
                optimizer.step()    
                optimizer.zero_grad()

            # accuracy
            _, predicted_indices = torch.max(y_pred, dim=1)
            _, label_indices = torch.max(labels, dim=1)
            n_total += y_pred.shape[0]
            n_correct += (predicted_indices == label_indices).sum().item()
            if (idx+1)%10==0:
                print(f'predicted_indices[0]:{predicted_indices[0]}, labels[0]:{label_indices[0]}')
                print(f'epoch:{epoch}, batch idx:{idx}, loss:{loss.item()}, accuracy:{n_correct / n_total}, n_total:{n_total}, n_correct:{n_correct}')
        
        # add summary    
        avg_train_loss = train_loss / len(train_loader)
        summary_writer.add_scalar('loss/train', avg_train_loss, epoch)
        
        # log param weights
        for name,param in cifar10_vit.named_parameters():
            summary_writer.add_histogram(name, param, epoch)
        
        # test
        with torch.no_grad():
            test_loss=0.0
            for batch_idx,(samples, labels) in enumerate(test_loader):
                samples=samples.to(device)
                labels=labels.to(device)
                y_pred=cifar10_vit(samples)
                test_loss+=criterion(y_pred,labels).item()
            # Log the average test loss for this epoch
            avg_test_loss = test_loss / len(test_loader)
            summary_writer.add_scalar('loss/test', avg_test_loss, epoch)
        
        # update learning rate
        scheduler.step()
        
        # save model
        torch.save(cifar10_vit.state_dict(), torch_model_path)
        
        t_end=time.time()
        print(f'elapsed time of one epoch is {t_end-t_start}')
    
    # close writer
    summary_writer.close()

if __name__ =="__main__":
    # img, label=train_dataset[34]
    # img=img.permute(1,2,0)
    # print(f'img.shape:{img.shape}, img type:{type(img)}, label.shape:{label}, label.type:{type(label)}')
    # plt.imshow(img)
    # plt.title(f'{label}')
    # plt.show()
    
    # train
    train()
