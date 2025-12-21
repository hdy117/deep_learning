# =============================================================================
# 基于 diffusers 库的条件扩散概率模型（Conditional DDPM）用于双色球彩票号码生成
# 使用 diffusers 库的调度器和管道，保持与 lot_ddpm.py 相同的功能
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import logging

from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.models import UNet1DModel
from diffusers.utils import BaseOutput
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 位置编码和条件编码器（使用transformers库）
# =============================================================================

class ConditionalEncoder(nn.Module):
    """
    条件编码器，使用transformers库的BertEncoder处理条件序列
    
    使用transformers库的BertEncoder和BertEmbeddings来实现条件编码，
    保持与原始实现相同的功能，但使用更标准化的transformer组件。
    """
    def __init__(self, input_dim=7, model_dim=128, num_layers=1, nhead=8):
        super().__init__()
        
        # 输入嵌入层：将原始特征映射到模型维度
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, model_dim),
        )
        
        # 使用transformers库的BertConfig配置编码器
        config = BertConfig(
            vocab_size=1,  # 不需要词汇表，仅用于配置
            hidden_size=model_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=nhead,
            intermediate_size=model_dim * 4,  # 前馈网络维度
            hidden_act='gelu',
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=model_dim + 1,  # 支持的最大序列长度
            type_vocab_size=1,
            initializer_range=0.02,
            _attn_implementation="eager",  # 设置注意力实现方式
        )
        
        # 使用transformers库的BertEmbeddings（包含位置编码）
        self.embeddings = BertEmbeddings(config)
        
        # 使用transformers库的BertEncoder
        self.encoder = BertEncoder(config)
        
        # 类别标记（class token），用于提取全局条件信息
        self.class_token = nn.Parameter(torch.randn(1, 1, model_dim))
        
        # 备用LayerNorm（如果BertEmbeddings没有LayerNorm属性时使用）
        self.fallback_layer_norm = nn.LayerNorm(model_dim)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 初始化class token
        nn.init.normal_(self.class_token, std=0.02)

    def forward(self, x):
        """
        前向传播，将条件序列编码为条件嵌入
        
        Args:
            x (torch.Tensor): 条件序列，形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 条件嵌入向量，形状为 [batch_size, model_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 将输入特征映射到模型维度
        x = self.embedding(x)  # [batch_size, seq_len, model_dim]
        
        # 添加class token到序列开头
        class_token = self.class_token.expand(batch_size, -1, -1)  # [batch_size, 1, model_dim]
        x = torch.cat([class_token, x], dim=1)  # [batch_size, seq_len+1, model_dim]
        
        # 创建位置ID用于位置编码
        seq_len_with_token = x.size(1)
        position_ids = torch.arange(
            seq_len_with_token, 
            dtype=torch.long, 
            device=x.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # 使用BertEmbeddings添加位置编码和层归一化
        # 由于BertEmbeddings期望input_ids，我们直接使用其位置编码和归一化组件
        position_embeddings = self.embeddings.position_embeddings(position_ids)  # 这是nn.Embedding，可以直接调用
        x = x + position_embeddings
        
        # 应用层归一化和dropout
        # 注意：不同版本的transformers可能使用不同的属性名（LayerNorm或layer_norm）
        if hasattr(self.embeddings, 'LayerNorm'):
            x = self.embeddings.LayerNorm(x)
        elif hasattr(self.embeddings, 'layer_norm'):
            x = self.embeddings.layer_norm(x)
        else:
            # 如果没有找到LayerNorm，使用备用的LayerNorm
            x = self.fallback_layer_norm(x)
        
        if hasattr(self.embeddings, 'dropout'):
            x = self.embeddings.dropout(x)
        
        # 创建attention mask（全1表示所有位置都参与注意力）
        attention_mask = torch.ones(
            (batch_size, seq_len_with_token), 
            dtype=torch.long, 
            device=x.device
        )
        
        # 扩展attention mask的维度以匹配BertEncoder的期望格式
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # 通过BertEncoder处理序列
        encoder_outputs = self.encoder(
            hidden_states=x,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # 返回class token的输出作为全局条件嵌入
        return encoder_outputs.last_hidden_state[:, 0]  # [batch_size, model_dim]

# =============================================================================
# 自定义 UNet1D 模型（兼容 diffusers）
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置嵌入，用于扩散模型的时间步编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * 
                       -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class LinearBlock(nn.Module):
    """线性残差块，用于UNet架构中的特征处理"""
    def __init__(self, im_dim, out_dim, embedding_dim=128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(im_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.embedding_proj = nn.Linear(embedding_dim, im_dim)
        self.short_cut = nn.Linear(im_dim, out_dim) if im_dim != out_dim else nn.Identity()
        
        self.out_activation = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x, embedding): 
        embedding_proj = self.embedding_proj(embedding)
        x = x + embedding_proj
        
        x_short = self.short_cut(x)
        x = self.block(x)
        x = x + x_short

        return self.out_activation(x)

class UNet1D(nn.Module):
    """一维UNet网络，用于扩散模型的噪声预测（兼容diffusers）"""
    def __init__(self, in_dim=7, base_dim=32, embedding_dim=128, num_cond_feature=7):
        super().__init__()
        
        self.in_dim = in_dim
        self.base_dim = base_dim
        self.embedding_dim = embedding_dim
        
        # 时间步嵌入网络
        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(embedding_dim),
            nn.Linear(embedding_dim, 4*embedding_dim),
            nn.GELU(),
            nn.Linear(4*embedding_dim, embedding_dim),
        )
        
        # 条件嵌入网络
        self.condition_embedding = ConditionalEncoder(
            input_dim=num_cond_feature, 
            model_dim=embedding_dim
        ) 
    
        # 初始投影层
        self.init_proj = nn.Sequential(
            nn.Linear(in_dim, base_dim),
            nn.GELU(),
        )
        
        # 下采样块
        self.down1 = LinearBlock(base_dim, base_dim * 2, embedding_dim=embedding_dim)
        self.down2 = LinearBlock(base_dim * 2, base_dim * 4, embedding_dim=embedding_dim)
        self.down3 = LinearBlock(base_dim * 4, base_dim * 8, embedding_dim=embedding_dim)

        # 瓶颈层
        self.bottleneck = LinearBlock(base_dim * 8, base_dim * 8, embedding_dim=embedding_dim)
        
        # 上采样块
        self.up1 = LinearBlock(base_dim * 16, base_dim * 4, embedding_dim=embedding_dim)
        self.up2 = LinearBlock(base_dim * 8, base_dim * 2, embedding_dim=embedding_dim)
        self.up3 = LinearBlock(base_dim * 4, base_dim * 1, embedding_dim=embedding_dim)

        # 输出投影层
        self.out_proj = nn.Sequential(
            nn.Linear(base_dim, in_dim), 
        )

    def forward(self, sample, timestep, encoder_hidden_states=None, return_dict=True):
        """
        前向传播，预测噪声（兼容diffusers接口）
        
        Args:
            sample: 带噪声的输入，形状为 [batch_size, in_dim]
            timestep: 时间步，形状为 [batch_size]
            encoder_hidden_states: 条件序列，形状为 [batch_size, sequence, condition_dim]
            return_dict: 是否返回字典格式
        """
        x = sample
        t = timestep
        
        # 初始投影
        x = self.init_proj(x)
        
        # 时间步嵌入
        t_embedding = self.t_embedding(t)
        
        # 条件嵌入
        if encoder_hidden_states is not None:
            cond_embedding = self.condition_embedding(encoder_hidden_states)
        else:
            cond_embedding = torch.zeros_like(t_embedding)
        
        # 融合嵌入
        embeddings = t_embedding + cond_embedding

        # 下采样路径
        x1 = self.down1(x, embeddings)
        x2 = self.down2(x1, embeddings)
        x3 = self.down3(x2, embeddings)
        
        # 瓶颈层
        x4 = self.bottleneck(x3, embeddings)

        # 上采样路径
        x = self.up1(torch.cat([x4, x3], dim=1), embeddings)
        x = self.up2(torch.cat([x, x2], dim=1), embeddings)
        x = self.up3(torch.cat([x, x1], dim=1), embeddings)

        # 输出投影
        noise_pred = self.out_proj(x)
        
        if return_dict:
            return UNet1DOutput(sample=noise_pred)
        else:
            return noise_pred

@dataclass
class UNet1DOutput(BaseOutput):
    """UNet1D的输出格式（兼容diffusers）"""
    sample: torch.FloatTensor

# =============================================================================
# 数据集类（保持原有实现）
# =============================================================================

class LotDataset(torch.utils.data.Dataset):
    """双色球彩票数据集类"""
    def __init__(self, data_path='./data/lot_data.csv', seq_length=72, out_dim=7, pre_scale=16.5, post_scale=8.0):
        super().__init__()
        self.data_path = data_path
        data_df = pd.read_csv(self.data_path)
        
        # Convert to numpy array for faster access (vectorized operations)
        # Extract all columns in order: red_ball_0 to red_ball_5, then blue_ball_0
        self.data = np.zeros((len(data_df), out_dim), dtype=np.float32)
        for ii in range(out_dim-1):
            self.data[:, ii] = data_df[f'red_ball_{ii}'].values
        self.data[:, out_dim-1] = data_df[f'blue_ball_0'].values
        
        self.seq_length = seq_length
        self.out_dim = out_dim
        self.pre_scale = pre_scale
        self.post_scale = post_scale
        
        self.dataset_length = len(self.data) - self.seq_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # Vectorized data loading: extract (seq_length+1) rows at once
        # This is much faster than looping with pandas iloc
        condition_data = self.data[idx:idx+self.seq_length+1]  # [seq_length+1, out_dim]
        condition = torch.from_numpy(condition_data).float()  # [seq_length+1, out_dim]
        
        # 标准化条件
        pre_cond = (condition[:, 0:self.out_dim-1] - self.pre_scale) / self.pre_scale
        post_cond = (condition[:, (self.out_dim-1):] - self.post_scale) / self.post_scale
        condition_scale = torch.cat((pre_cond, post_cond), dim=1)
        
        # 提取x0
        x0 = condition_scale[self.seq_length:, :].squeeze(0)
        
        # 提取条件
        condition_scale = condition_scale[0:self.seq_length, :]
        
        condition_scale = condition_scale.to(torch.float)
        x0 = x0.to(torch.float)
                    
        return condition_scale, x0

# =============================================================================
# 配置类
# =============================================================================

class Config:
    """模型配置类，包含所有训练和推理参数"""
    def __init__(self):
        # 数据相关配置
        self.data_path = './data/lot_data.csv'
        self.model_path = './models/ddpm_lot_diffusers.pth'
        
        # 设备配置
        self.device = DEVICE
        
        # 数据标准化参数
        self.pre_scale = 16.5
        self.post_scale = 8.0
        
        # 模型架构参数
        self.cond_seq_lenth = 72
        self.condition_feature_dim = 7
        self.out_dim = 7
        self.ddpm_scheduler_steps = 1000
        
        # 初始化UNet模型
        self.unet = UNet1D(
            in_dim=self.out_dim, 
            base_dim=64, 
            embedding_dim=256, 
            num_cond_feature=self.condition_feature_dim
        ).to(self.device)
        
        # 初始化diffusers调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.ddpm_scheduler_steps,
            beta_schedule="linear",
            beta_start=1e-4,
            beta_end=0.02,
            prediction_type="epsilon",  # 预测噪声
        )
        
        # 数据集和数据加载器
        self.dataset = LotDataset(
            data_path=self.data_path, 
            seq_length=self.cond_seq_lenth, 
            out_dim=self.out_dim,
            pre_scale=self.pre_scale, 
            post_scale=self.post_scale
        )
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, 
            batch_size=128, 
            shuffle=True
        )
        
        # 训练配置
        self.lr = 1e-4
        self.epochs = 1637
        self.optimizer = torch.optim.Adam(
            self.unet.parameters(), 
            lr=self.lr, 
            weight_decay=1e-5
        )
        self.criterion = nn.MSELoss()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-5
        )
        
        # 采样配置
        self.sample_batch_size = 10
        self.out_file = './lot_ddpm_diffusers.txt'
        
        # CFG引导比例
        self.guidance_scale = 3.0

# =============================================================================
# 训练和推理函数
# =============================================================================

def train():
    """训练DDPM模型（使用diffusers调度器）"""
    config = Config()
    unet = config.unet
    noise_scheduler = config.noise_scheduler
    losses = []
    best_loss = 1e9
    
    # 加载预训练模型（如果存在）
    if os.path.exists(config.model_path):
        unet.load_state_dict(torch.load(config.model_path))
        unet.train()
        logging.info(f'UNet model loaded from {config.model_path}')
    
    # 训练循环
    for epoch_i in range(config.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(config.data_loader, desc=f"Epoch {epoch_i+1}/{config.epochs}")
        
        for one_batch in progress_bar:
            # 清零梯度
            config.optimizer.zero_grad()
            
            # 获取条件序列和真实数据
            condition = one_batch[0].to(config.device)  # [batch_size, sequence, condition_feature_dim]
            x0 = one_batch[1].to(config.device)         # [batch_size, out_dim]
            
            # 随机选择时间步
            batch_size = condition.shape[0]
            timesteps = torch.randint(
                0, 
                noise_scheduler.config.num_train_timesteps, 
                (batch_size,), 
                device=condition.device
            ).long()
            
            # 生成噪声
            noise = torch.randn_like(x0)
            
            # 使用调度器添加噪声
            noisy_samples = noise_scheduler.add_noise(x0, noise, timesteps)
            
            # 随机丢弃条件（提高鲁棒性）
            keep_mask = (torch.rand(batch_size, device=condition.device) > 0.2).float()[:, None, None]
            condition = condition * keep_mask
            
            # 使用UNet预测噪声
            model_output = unet(
                sample=noisy_samples,
                timestep=timesteps,
                encoder_hidden_states=condition,
                return_dict=True
            ).sample
            
            # 计算损失
            loss = config.criterion(model_output, noise)
            loss.backward()
            config.optimizer.step()
            
            # 更新进度条和损失统计
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # 更新学习率
        config.lr_scheduler.step()
        
        # 定期保存模型
        if (epoch_i+1) % 10 == 0:
            torch.save(unet.state_dict(), config.model_path)
            logging.info(f'Model saved to {config.model_path}')
        
        # 记录平均损失
        avg_loss = epoch_loss / len(config.data_loader)
        losses.append(avg_loss)
        logging.info(f"Epoch {epoch_i+1}/{config.epochs}, Average Loss: {avg_loss:.6f}")
    
    # 绘制并保存训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Diffusers)')
    plt.savefig('./lot_ddpm_diffusers_loss_curve.png')
    plt.close()

def append_list_to_file(file_path, items_list):
    """将列表追加到文件中"""
    try:
        with open(file_path, 'a') as file:
            file.write(f"{items_list}\n")
    except IOError as e:
        print(f"An error occurred: {e}")

def sample():
    """从训练好的DDPM模型生成彩票号码（使用diffusers调度器）"""
    config = Config()
    unet = config.unet
    noise_scheduler = config.noise_scheduler
    
    # 加载训练好的模型
    if os.path.exists(config.model_path):
        unet.load_state_dict(torch.load(config.model_path))
        unet.eval()
        logging.info(f'UNet model loaded from {config.model_path}')
    else:
        logging.info(f'No model found at {config.model_path}, please train the model first.')
        return

    with torch.no_grad():
        unet.to(config.device)
        
        # 加载最后一个条件
        condition, x0 = config.dataset[config.dataset.__len__()-1]
        condition = torch.cat([condition[1::, :], x0.unsqueeze(0)], dim=0)
        condition = condition.to(config.device)
        
        # 扩展条件以匹配采样批次大小
        condition = condition.expand(config.sample_batch_size, -1, -1)
        
        # 从纯噪声开始
        shape = (config.sample_batch_size, config.out_dim)
        noise = torch.randn(shape, device=config.device)
        
        # 设置调度器为推理模式
        noise_scheduler.set_timesteps(config.ddpm_scheduler_steps)
        
        # 反向去噪过程
        latents = noise
        for t in noise_scheduler.timesteps:
            # 为CFG准备：将批次大小翻倍
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.repeat(2)
            condition_concat = torch.cat([condition, torch.zeros_like(condition)], dim=0)
            
            # 预测噪声
            noise_pred = unet(
                sample=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=condition_concat,
                return_dict=True
            ).sample
            
            # 分离条件和无条件预测
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            
            # 应用CFG
            if config.guidance_scale > 1.0:
                noise_pred = noise_pred_cond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            
            # 使用调度器进行去噪步骤
            latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # 后处理：反标准化和格式化
        samples = latents.cpu()
        for batch_i in range(samples.shape[0]):
            sample = samples[batch_i]
            sample = sample.view(config.out_dim)
            
            # 反标准化
            pre_sample = (sample[0:(config.out_dim-1)] + 1.0) * config.pre_scale
            post_sample = (sample[(config.out_dim-1):] + 1.0) * config.post_scale
            
            # 转换为整数并裁剪到有效范围
            pre_sample = torch.clip(pre_sample.to(torch.int), 1, int(2*config.pre_scale))
            post_sample = torch.clip(post_sample.to(torch.int), 1, int(2*config.post_scale))
            
            # 合并红球和蓝球
            sample = torch.cat((pre_sample, post_sample), dim=0)
            sample = sample.tolist()
            
            # 检查重复：只有没有重复号码的样本才被接受
            sample_set = set(sample)
            if len(sample_set) == config.out_dim:
                print(f'{sample}')
                append_list_to_file(config.out_file, sample)

# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    """
    主函数：程序入口点
    
    使用方法：
    python lot_ddpm_diffusers.py --train    # 训练模型
    python lot_ddpm_diffusers.py --sample   # 生成彩票号码
    """
    
    # 配置日志系统
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s[%(levelname)s][%(filename)s][%(lineno)s] - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 解析命令行参数
    args_parser = argparse.ArgumentParser(description="Train or sample from the Lot DDPM model (Diffusers)")
    args_parser.add_argument('--train', action='store_true', help='Train the DDPM model')
    args_parser.add_argument('--sample', action='store_true', help='Sample from the DDPM Model')
    args = args_parser.parse_args()
    
    # 根据参数执行相应功能
    if args.train:
        logging.info(f'Training the DDPM model with diffusers...')
        train()
    elif args.sample:  
        logging.info(f'Sampling from the DDPM model with diffusers...')     
        sample()
    else:
        logging.info("Please specify --train or --sample to run the script.")
        args_parser.print_help()
        exit(1)

