import torch
import torch.nn as nn
import math
import torchvision
import matplotlib.pyplot as plt
from cifar10_dataset import *

class DDPM(nn.Module):
    def __init__(self, T=10, beta_start=1e-4, beta_end=2e-2,device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(DDPM, self).__init__()
        self.device=device
        self.T=T
        self.beta_start=beta_start
        self.beta_end=beta_end
        self.betas=torch.linspace(self.beta_start, self.beta_end, self.T, device=self.device)
        self.alpha = 1 - self.betas
        self.alpha_bar=torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar=torch.sqrt(self.alpha_bar)
        self.sqrt_one_minux_alpha_bar=torch.sqrt(1-self.alpha_bar)
        
    
    def ddpm_forward_step(self, x:torch.Tensor, t:int=0):
        '''
        DDPM forward process
        Args:
            x (torch.Tensor): 输入图像张量，形状为 [batch, channel, height, width]
            t (int): 当前时间步, 默认为0
        '''
        # print(f"ddpm_forward: x.shape={x.shape}, x.device={x.device}")
        # print(f'ddpm_forward: betas={self.betas}')
        # print(f'ddpm_forward: alpha={self.alpha}')
        # print(f'ddpm_forward: alpha_bar={self.alpha_bar}')
        
        eps=torch.randn_like(x, device=self.device)  # generate noise
        x_t=self.sqrt_alpha_bar[t]*x+self.sqrt_one_minux_alpha_bar[t]*eps  # add noise
        
        return x_t, eps  # return image with noise and noise itself

    def ddpm_forward(self, x:torch.Tensor):
        '''
        DDPM forward process for a batch of images
        Args:
            x (torch.Tensor): 输入图像张量，形状为 [batch, channel, height, width]
        Returns:
            x_out (torch.Tensor): 添加噪声后的图像张量，形状为 [batch, T, channel, height, width]
        '''
        x_t_all=[]
        for t in range(self.T):
            x_t, eps = self.ddpm_forward_step(x, t)
            x_t_all.append(x_t.unsqueeze(1)) # add step t dimension
            # print(f"Step {t+1}/{self.T}: x.shape={x.shape}, eps.shape={eps.shape}")
        x_t_all= torch.cat(x_t_all, dim=1)  # concatenate along step dimension, [batch, T, channel, height, width]
        return x_t_all
        
        
# 测试代码
if __name__ == "__main__":
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    images = images.to(device)  # 将图像移动到DDPM所在的设备
    labels = labels.to(device)  # 将标签移动到DDPM所在的设备
    
    # ddpm forward process, adding nosise to the images
    T=15
    ddpm = DDPM(T=T, device=device)
    print(f"DDPM model initialized with T={ddpm.T}, beta_start={ddpm.beta_start}, beta_end={ddpm.beta_end}")
    images=ddpm.ddpm_forward(images) # [batch, T, channel, height, width]
    print(f'images.shape={images.shape}, images.device={images.device}')

    # 反标准化以显示原始图像
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(3, 1, 1)
    images = (images * std + mean).clip(0,1)  # 还原至 [0, 1] 范围
    images = images.cpu().numpy().transpose((0, 1, 3, 4, 2))  # 转换为 (batch_size, T, H, W, C) 格式

    # draw images
    # 可视化：每个样本的T个时间步图像
    batch_size=5
    plt.figure(figsize=(2*T, 2*batch_size))
    for i in range(batch_size):
        for t in range(T):
            ax = plt.subplot(batch_size, T, i*T + t + 1)
            plt.imshow(images[i, t])
            plt.title(f"t={t}\n{train_dataset.classes[labels[i]]}")
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('./ddpm_forward_images.png', dpi=300)
    plt.show()