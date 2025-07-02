import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
import argparse

# 设置随机种子以确保结果可复现
# torch.manual_seed(42)
# np.random.seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将数据标准化到[-1, 1]范围
])

# 加载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义时间嵌入模块
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim=dim
        if not(self.dim%2)==0:
            raise ValueError(f'{self.dim} is not an even number')
    
    def forward(self, time):
        '''
        time:[batch_size]
        '''
        half_dim=self.dim//2 # half dim
        omega=torch.exp(-1.0*math.log(10000.0)*(torch.arange(0,half_dim,device=time.device)*1.0/(half_dim-1))) # 1/(10000**i/d) # [half_dim]
        embedding=time[:,None]*omega[None,:] # [batch_size, half_dim]
        embedding=torch.concat([embedding.sin(), embedding.cos()],dim=-1) # [batch_size, dim]

        return embedding

class DoubleConv(nn.Module):
    """(Conv => BN => GELU) * 2"""
    def __init__(self, in_channels, out_channels,time_emb_dim,label_emb_dim):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, in_channels)
        self.label_mlp = nn.Linear(label_emb_dim, in_channels)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8,out_channels),
            nn.GELU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8,out_channels),
            nn.GELU()
        )
        self.gelu=nn.GELU()

    def forward(self, x, t_embeding, label_embeding):
        # add time/label embedding
        time_emb = self.time_mlp(t_embeding)
        label_emb=self.label_mlp(label_embeding)

        # adjust dim of time embedding to use broadcast of torch
        time_emb = time_emb[...,None,None] # [batch_size, time_emb_dim] --> [batch_size, time_emb_dim, 1, 1]
        label_emb=label_emb[...,None,None] # [batch_size, time_emb_dim] --> [batch_size, label_emb_dim, 1, 1]

        # add time/label embedding to input
        x=x+time_emb+label_emb
        
        # 应用双卷积层
        h = self.double_conv(x)
        
        return h

class UPSampleBlock(nn.Module):
    def __init__(self, feature_dim, time_emb_dim, label_emb_dim):
        super().__init__()
        self.tran_conv2d=nn.ConvTranspose2d(feature_dim*2, feature_dim, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(feature_dim*2, feature_dim, time_emb_dim, label_emb_dim)
        self.feature_dim=feature_dim
    
    def forward(self, x, skip_connection, t, label):
        # print(f'UPSampleBlock before tran_conv2d: x.shape={x.shape}, skip_connection.shape={skip_connection.shape}, self.feature_dim={self.feature_dim}')
        x=self.tran_conv2d(x)
        x= torch.cat([x, skip_connection], dim=1)  # 拼接跳跃连接
        x=self.double_conv(x, t, label)
        return x

# 定义完整的U-Net模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_dims=[64,128,256,512], time_emb_dim=256, label_emb_dim=256):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, 4*time_emb_dim),
            nn.GELU(),
            nn.Linear(4*time_emb_dim, time_emb_dim),
        )

        self.label_mlp = nn.Sequential(
            nn.Embedding(10,label_emb_dim),
            nn.GELU(),
            nn.Linear(label_emb_dim,label_emb_dim),
        )

        hidden_channels=feature_dims[0]
        feature_dims=feature_dims[1:]
        
        # init conv
        self.init_conv=nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8,hidden_channels),
            nn.GELU(),
        )
        
        # down sample blocks
        self.down_smaple_blocks=nn.ModuleList()
        
        # up sample blocks
        self.up_sample_blocks=nn.ModuleList()

        self.label_embedding_dim=label_emb_dim
        
        # adding down/up sample blocks
        feature_dim_in=hidden_channels
        for feataure in feature_dims:
            self.down_smaple_blocks.append(DoubleConv(feature_dim_in,feataure,time_emb_dim,label_emb_dim))
            feature_dim_in=feataure
        
        for feature in reversed(feature_dims):
            self.up_sample_blocks.append(UPSampleBlock(feature, time_emb_dim,label_emb_dim))
        
        # bottle neck
        self.bottle_neck=DoubleConv(feature_dims[-1],feature_dims[-1]*2,time_emb_dim, label_emb_dim)
        
        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(feature_dims[0], out_channels, kernel_size=3, padding=1),
            # nn.Tanh(),
        )
        
        # down sample max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, time, label):
        '''
        x:[batch_size,C,W,H],
        time:[batch_size]
        label:[batch_size]
        '''
        # 时间嵌入
        t = self.time_mlp(time)

        # label embedding
        if label is None or (label == -1).all():
            label_emb = torch.zeros(x.shape[0], self.label_embedding_dim, device=x.device)
        else:
            label_emb=self.label_mlp(label)
        
        # init conv
        x=self.init_conv(x)  # 初始卷积层
        
        #skip connections
        skip_connections = []
        
        # down sampling
        for bolck in self.down_smaple_blocks:
            x = bolck(x, t, label_emb)
            skip_connections.append(x)
            x=self.pool(x)  # 下采样
        
        skip_connections=list(reversed(skip_connections))  # 反转以便在上采样时使用
        x=self.bottle_neck(x, t, label_emb)  # 瓶颈层
        
        # up sampling
        for i, bock in enumerate(self.up_sample_blocks):
            x = bock(x,skip_connections[i],t, label_emb)
        
        # 输出层
        return self.out(x)

# =====================
# Utility: Cosine beta schedule
# =====================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    线性 beta schedule,从 beta_start 线性递增到 beta_end。
    
    Args:
        timesteps (int): 扩散步骤数，比如 1000。
        beta_start (float): beta 起始值。
        beta_end (float): beta 结束值。
    
    Returns:
        torch.Tensor: 长度为 timesteps 的 beta 序列。
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# 定义DDPM模型
class DDPM(nn.Module):
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, num_diffusion_timesteps=1000):
        super().__init__()
        self.unet = model
        self.num_diffusion_timesteps = num_diffusion_timesteps
        
        # for forward and generate process
        # betas = cosine_beta_schedule(self.num_diffusion_timesteps)
        betas = linear_beta_schedule(self.num_diffusion_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("posterior_variance", betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))
        
    def forward_diffusion(self, x0, t, noise=None):
        """正向扩散过程:直接从x_0计算x_t"""
        if noise is None:
            noise=torch.rand_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def p_loss(self, x0, t, label, p_null=0.1):
        noise = torch.randn_like(x0)
        x_t, noise = self.forward_diffusion(x0, t, noise)
        # use_null = (torch.rand(x0.shape[0], device=x0.device) < p_null)
        # label_input = label.clone()
        # label_input[use_null] = -1
        pred_noise = self.unet(x_t, t, label)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, shape, label=None, guidance_scale=3.0):
        device = next(self.parameters()).device
        img = torch.randn(shape, device=device)
        sample_step=100
        sample_steps=[]
        for i in reversed(range(self.num_diffusion_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            if guidance_scale == 1.0 or label is None:
                noise_pred = self.unet(img, t, label)
            else:
                noise_pred_cond = self.unet(img, t, label)
                noise_pred_uncond = self.unet(img, t, None)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1, 1)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

            mean = (1 / torch.sqrt(alpha_t)) * (img - beta_t * noise_pred / sqrt_one_minus)

            if i > 0:
                noise = torch.randn_like(img)
                var = self.posterior_variance[t].view(-1, 1, 1, 1)
                img = mean + torch.sqrt(var) * noise
            else:
                img = mean
            
            # save sample steps results
            if i%sample_step==0:
                sample_steps.append(img[0].detach().cpu())
        return img,sample_steps

# 训练函数
def train_ddpm(model, dataloader, optimizer, scheduler, num_epochs, device, save_dir='./models'):
    """训练DDPM模型"""
    os.makedirs(save_dir, exist_ok=True)
    
    # try to load existing model
    if os.path.exists(f"{save_dir}/ddpm_epoch.pt"):
        print("加载已有模型...")
        model.load_state_dict(torch.load(f"{save_dir}/ddpm_epoch.pt", map_location=device))
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # 获取数据
            x_0 = batch[0].to(device)
            label = batch[1].to(device)
            batch_size = x_0.shape[0]
            
            # 随机采样时间步
            t = torch.randint(0, model.num_diffusion_timesteps, (batch_size,), device=device).long()
            
            # 计算损失
            loss = model.p_loss(x_0, t, label)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # update learning rate
        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # 每个epoch保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{save_dir}/ddpm_epoch.pt")
        
        # 每5个epoch生成一些样本
        with torch.no_grad():
            if (epoch + 1) % 5 == 0:
                generate_samples(epoch=epoch,device=device,n_samples=16,dataset=train_dataset)
    
    torch.save(model.state_dict(), f"{save_dir}/ddpm_epoch.pt")
    
    return losses

# 生成样本函数
def generate_samples(epoch, device, dataset:torch.utils.data.Dataset, n_samples=16, save_dir='./samples'):
    """从DDPM模型生成样本并保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化模型
    unet = UNet(in_channels=3, out_channels=3, feature_dims=[64,128,256,512]).to(device)
    ddpm = DDPM(model=unet, num_diffusion_timesteps=1000).to(device)

    if os.path.exists(f"./models/ddpm_epoch.pt"):
        ddpm.load_state_dict(torch.load(f"./models/ddpm_epoch.pt",map_location=device))
    else:
        raise ValueError(f'fail to load ./models/ddpm_epoch.pt')

    shape=(16,3,32,32)
    label=torch.randint(0,10,(16,)).to(device)
    guidance_scale=1.0

    ddpm.eval()
    samples, sample_steps = ddpm.sample(shape,label,guidance_scale)
    
    # save_images(samples, sample_steps, n_samples, epoch, save_dir)
    # def save_images(samples, sample_steps, n_samples=16, epoch=10, save_dir='./samples'):
    # 保存最终样本
    plt.figure(figsize=(16, 16))
    for i in range(min(16, n_samples)):
        plt.subplot(4, 4, i+1)
        plt.title(f'{dataset.classes[label[i]]}, label:{label[i]}')
        img = samples[i].cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2  # 从[-1,1]范围转换到[0,1]范围
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/samples_epoch_{epoch}.png")
    plt.close()
    
    # 保存去噪过程
    plt.figure(figsize=(15, 5))
    for i, step in enumerate(sample_steps):
        plt.subplot(1, len(sample_steps), i+1)
        img = step.cpu().numpy().transpose(1, 2, 0)
        img = (img + 1) / 2  # 从[-1,1]范围转换到[0,1]范围
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        plt.title(f"Step {i}")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/denoising_process_epoch_{epoch}.png")
    plt.close()


# 主函数
def main(args):
    if not args.sample:
        # 初始化模型
        unet = UNet(in_channels=3, out_channels=3, feature_dims=[64,128,256,512]).to(device)
        ddpm = DDPM(model=unet, num_diffusion_timesteps=1000).to(device)
        
        num_epochs = 10

        # 定义优化器
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=10,eta_min=2e-5)
        
        # 训练模型
        losses = train_ddpm(ddpm, train_dataloader, optimizer, scheduler=scheduler, num_epochs=num_epochs, device=device)
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('./loss_curve.png')
        plt.close()
    
    # 生成最终样本
    generate_samples(epoch='final', device=device, n_samples=16, dataset=train_dataset)

if __name__ == "__main__":
    arg_parser=argparse.ArgumentParser(description='ddpm console')
    arg_parser.add_argument('--sample','-s',type=bool,default=False,help='true: sample only, false: train and sample')
    args=arg_parser.parse_args()
    main(args)
