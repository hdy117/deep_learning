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

# 设置随机种子以确保结果可复现
# torch.manual_seed(42)
# np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义时间嵌入模块
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        
        # 如果输入输出通道数不同，添加跳跃连接
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        # 应用第一个卷积层
        h = self.gelu(self.bnorm1(self.conv1(x)))
        
        # 添加时间嵌入
        time_emb = self.gelu(self.time_mlp(time_emb))
        # 调整时间嵌入的维度以匹配特征图
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        
        # 应用第二个卷积层
        h = self.bnorm2(self.conv2(h))
        
        # 添加残差连接
        return self.gelu(h + self.residual(x))

class DoubleConv(nn.Module):
    """(Conv => GELU => BN) * 2"""
    def __init__(self, in_channels, out_channels,time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.gelu=nn.GELU()

    def forward(self, x, t):
        # 添加时间嵌入
        time_emb = self.gelu(self.time_mlp(t))
        # 调整时间嵌入的维度以匹配特征图
        time_emb = time_emb[(..., ) + (None, ) * 2]
        
        # 应用双卷积层
        h = self.double_conv(x)
        x=h+time_emb
        return self.gelu(h)

class UPSampleBlock(nn.Module):
    def __init__(self, feature_dim, time_emb_dim):
        super().__init__()
        self.tran_conv2d=nn.ConvTranspose2d(feature_dim*2, feature_dim, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(feature_dim*2, feature_dim, time_emb_dim)
        self.feature_dim=feature_dim
    
    def forward(self, x, skip_connection, t):
        # print(f'UPSampleBlock before tran_conv2d: x.shape={x.shape}, skip_connection.shape={skip_connection.shape}, self.feature_dim={self.feature_dim}')
        x=self.tran_conv2d(x)
        x= torch.cat([x, skip_connection], dim=1)  # 拼接跳跃连接
        x=self.double_conv(x, t)
        return x

# 定义完整的U-Net模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=64, time_emb_dim=256):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )
        
        # init conv
        self.init_conv=nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        
        # down sample blocks
        self.down_smaple_blocks=nn.ModuleList()
        
        # up sample blocks
        self.up_sample_blocks=nn.ModuleList()
        
        # adding down/up sample blocks
        feature_dims=[128,256,512]
        feature_dim_in=hidden_channels
        for feataure in feature_dims:
            self.down_smaple_blocks.append(DoubleConv(feature_dim_in,feataure,time_emb_dim))
            feature_dim_in=feataure
        
        for feature in reversed(feature_dims):
            self.up_sample_blocks.append(UPSampleBlock(feature, time_emb_dim))
        
        # bottle neck
        self.bottle_neck=DoubleConv(feature_dims[-1],feature_dims[-1]*2,time_emb_dim)
        
        # 输出层
        self.out = nn.Conv2d(feature_dims[0], out_channels, 1)
        
        # down sample max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, time):
        # 时间嵌入
        t = self.time_mlp(time)
        
        # init conv
        x=self.init_conv(x)  # 初始卷积层
        
        #skip connections
        skip_connections = []
        
        # down sampling
        for bolck in self.down_smaple_blocks:
            x = bolck(x, t)
            skip_connections.append(x)
            x=self.pool(x)  # 下采样
        
        skip_connections=list(reversed(skip_connections))  # 反转以便在上采样时使用
        x=self.bottle_neck(x, t)  # 瓶颈层
        
        # up sampling
        for i, bock in enumerate(self.up_sample_blocks):
            x = bock(x,skip_connections[i],t)
        
        # 输出层
        return self.out(x)

# 定义DDPM模型
class DDPM(nn.Module):
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, num_diffusion_timesteps=1000):
        super().__init__()
        self.model = model
        self.num_diffusion_timesteps = num_diffusion_timesteps
        
        # 线性调度的噪声系数
        self.betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 预计算一些常用值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def forward_diffusion(self, x_0, t, noise=None):
        """正向扩散过程:直接从x_0计算x_t"""
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # 公式: x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1-alphas_cumprod_t) * noise
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def p_loss(self, x_0, t):
        """训练损失:预测噪声并计算MSE损失"""
        noise = torch.randn_like(x_0)
        x_t, noise = self.forward_diffusion(x_0, t, noise)
        
        # 模型预测噪声
        predicted_noise = self.model(x_t, t)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, x_t, t):
        """从x_t采样x_{t-1}（单步去噪）"""
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, x_t.shape)
        
        # 预测噪声
        predicted_noise = self.model(x_t, t)
        
        # 公式: x_{t-1} = sqrt(1/beta_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t) * predicted_noise) + sqrt(variance) * noise
        mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] > 0:  # 只有当t>0时才添加噪声
            noise = torch.randn_like(x_t)
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            sample = mean + torch.sqrt(posterior_variance_t) * noise
        else:  # t=0时直接返回均值
            sample = mean
            
        return sample
    
    @torch.no_grad()
    def p_sample_loop(self, shape, device):
        """完整的去噪过程:从随机噪声开始，逐步生成样本"""
        b = shape[0]
        # 从随机噪声开始
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in tqdm(reversed(range(0, self.num_diffusion_timesteps)), desc='Sampling'):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
            if (i + 1) % 100 == 0 or i == 0:  # 每100步保存一次中间结果
                imgs.append(img.cpu().numpy())
            
        return img, imgs
    
    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3, device='cuda'):
        """生成新样本"""
        return self.p_sample_loop((batch_size, channels, image_size, image_size), device)
    
    def _extract(self, a, t, x_shape):
        """从张量a中提取与时间步t对应的元素"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# 训练函数
def train_ddpm(model, dataloader, optimizer, num_epochs, device, save_dir='./models'):
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
            batch_size = x_0.shape[0]
            
            # 随机采样时间步
            t = torch.randint(0, model.num_diffusion_timesteps, (batch_size,), device=device).long()
            
            # 计算损失
            loss = model.p_loss(x_0, t)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # 每个epoch保存模型
        # torch.save(model.state_dict(), f"{save_dir}/ddpm_epoch_{epoch+1}.pt")
        
        # 每5个epoch生成一些样本
        # if (epoch + 1) % 5 == 0:
        #     generate_samples(model, epoch+1, device)
    
    torch.save(model.state_dict(), f"{save_dir}/ddpm_epoch.pt")
    
    return losses

# 生成样本函数
def generate_samples(model, epoch, device, n_samples=16, save_dir='./samples'):
    """从DDPM模型生成样本并保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    samples, sample_steps = model.sample(image_size=32, batch_size=n_samples, device=device)
    
    # 保存最终样本
    plt.figure(figsize=(10, 10))
    for i in range(min(16, n_samples)):
        plt.subplot(4, 4, i+1)
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
        img = step[0].transpose(1, 2, 0)
        img = (img + 1) / 2  # 从[-1,1]范围转换到[0,1]范围
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        plt.title(f"Step {step.shape[0] - i - 1}")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/denoising_process_epoch_{epoch}.png")
    plt.close()

# 主函数
def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将数据标准化到[-1, 1]范围
    ])
    
    # 加载CIFAR-10数据集
    train_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    
    # 初始化模型
    unet = UNet(in_channels=3, out_channels=3, hidden_channels=64).to(device)
    ddpm = DDPM(model=unet, num_diffusion_timesteps=1000).to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-4,weight_decay=1e-3)
    
    # 训练模型
    losses = train_ddpm(ddpm, train_dataloader, optimizer, num_epochs=5, device=device)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('./loss_curve.png')
    plt.close()
    
    # 生成最终样本
    generate_samples(ddpm, 'final', device, n_samples=64)

if __name__ == "__main__":
    main()
