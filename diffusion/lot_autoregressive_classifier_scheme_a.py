import argparse
import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm


# 运行设备与脚本路径。
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# 固定随机种子，尽量保证训练与采样可复现。
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


def configure_torch_runtime(device, deterministic=False, enable_tf32=True):
    """按运行模式切换 PyTorch 后端配置。"""
    if device.type != 'cuda':
        return

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32


class PositionalEncoding(nn.Module):
    """标准正弦位置编码。"""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encoding', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)


class HistoryEncoder(nn.Module):
    """
    对过去 72 期历史号码做条件编码。

    输入 history 形状为 [B, seq_length, 7]：
    - 前 6 个位置为红球，范围 1..33
    - 最后 1 个位置为蓝球，范围 1..16

    编码思路：
    1. 红球和蓝球使用不同 embedding；
    2. 再叠加 field embedding，告诉模型当前号码来自 red_ball_0..5 还是 blue_ball_0；
    3. 每一期 7 个号码聚合成一个 draw token；
    4. 对 72 个 draw token 使用 TransformerEncoder 编码。
    """

    def __init__(self, d_model=128, nhead=8, num_layers=2, dropout=0.1, seq_length=72):
        super().__init__()
        self.red_embedding = nn.Embedding(34, d_model)
        self.blue_embedding = nn.Embedding(17, d_model)
        self.field_embedding = nn.Embedding(7, d_model)
        self.draw_positional_encoding = PositionalEncoding(d_model, max_len=seq_length + 1)
        self.register_buffer('red_field_ids', torch.arange(6, dtype=torch.long).view(1, 1, 6))
        self.register_buffer('blue_field_ids', torch.full((1, 1), 6, dtype=torch.long))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, history):
        # 历史输入是原始离散号码，不做连续标准化。
        red_values = history[:, :, :6]
        blue_values = history[:, :, 6]

        # 每个字段一个固定 field id，用于标识红球位置或蓝球位置。
        red_field_ids = self.red_field_ids
        blue_field_ids = self.blue_field_ids

        # 红球和蓝球分别走各自 embedding，再叠加字段 embedding。
        red_emb = self.red_embedding(red_values) + self.field_embedding(red_field_ids)
        blue_emb = self.blue_embedding(blue_values).unsqueeze(2) + self.field_embedding(blue_field_ids).unsqueeze(0)

        # 将单期 7 个号码聚合成一个向量；这里采用简单均值聚合，保持实现稳定直接。
        draw_emb = torch.cat([red_emb, blue_emb], dim=2).mean(dim=2)
        draw_emb = self.draw_positional_encoding(draw_emb)
        draw_emb = self.dropout(draw_emb)

        # 输出所有历史 token 和记忆池化向量。
        history_memory = self.transformer(draw_emb)
        history_memory = self.norm(history_memory)
        history_context = history_memory.mean(dim=1)
        return history_memory, history_context


class AutoregressiveLotteryModel(nn.Module):
    """
    基于 Transformer 的条件自回归分类模型。

    统一 token 词表：
    - 0 = BOS
    - 1..33 = red_1..33
    - 34..49 = blue_1..16

    解码阶段按顺序预测 7 个位置：
    1. red_ball_0
    2. red_ball_1
    3. red_ball_2
    4. red_ball_3
    5. red_ball_4
    6. red_ball_5
    7. blue_ball_0

    训练时使用 teacher forcing：
    - 输入 decoder prefix = BOS + 已知前序 token
    - 输出每个位置的分类 logits
    """

    def __init__(self, d_model=128, nhead=8, num_layers=3, dropout=0.1, seq_length=72):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.vocab_size = 50
        self.bos_token_id = 0
        self.register_buffer('causal_mask', torch.triu(torch.full((8, 8), float('-inf')), diagonal=1))

        self.history_encoder = HistoryEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            seq_length=seq_length,
        )

        # 目标序列 token embedding，与历史编码器分离。
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.token_positional_encoding = PositionalEncoding(d_model, max_len=8)

        # 将全局历史上下文投影后加到 decoder token 上，给解码器提供全局条件。
        self.context_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 使用 TransformerDecoder 做自回归解码。
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(d_model)

        # 红球和蓝球分开分类头，保持任务定义清晰。
        self.red_head = nn.Linear(d_model, 33)
        self.blue_head = nn.Linear(d_model, 16)

    @staticmethod
    def red_values_to_tokens(target_red):
        """将红球标签 0..32 映射为统一词表中的 1..33。"""
        return target_red + 1

    @staticmethod
    def blue_values_to_tokens(target_blue):
        """将蓝球标签 0..15 映射为统一词表中的 34..49。"""
        return target_blue + 34

    def build_decoder_inputs(self, target_red, target_blue):
        """
        teacher forcing 输入：
        decoder_input = [BOS, y1, y2, ..., y6]
        用它去预测 [y1, y2, ..., y7]。
        """
        red_tokens = self.red_values_to_tokens(target_red)
        blue_token = self.blue_values_to_tokens(target_blue).unsqueeze(1)
        target_tokens = torch.cat([red_tokens, blue_token], dim=1)

        bos = torch.full(
            (target_tokens.size(0), 1),
            self.bos_token_id,
            dtype=torch.long,
            device=target_tokens.device,
        )
        return torch.cat([bos, target_tokens[:, :-1]], dim=1)

    def encode_history(self, history):
        return self.history_encoder(history)

    def get_causal_mask(self, seq_len):
        """返回缓存的 decoder causal mask。"""
        return self.causal_mask[:seq_len, :seq_len]

    def decode_from_encoded(self, history_memory, history_context, decoder_input_tokens):
        """
        给定历史编码结果与 prefix token，输出 prefix 每个位置的隐藏状态。

        这里使用：
        - causal self-attention：保证自回归性质
        - cross-attention：读取历史序列 memory
        """
        decoder_input = self.token_embedding(decoder_input_tokens)
        decoder_input = self.token_positional_encoding(decoder_input)

        # 将 pooled history_context 广播到每个 target token 上，作为全局条件偏置。
        decoder_input = decoder_input + self.context_projection(history_context).unsqueeze(1)

        causal_mask = self.get_causal_mask(decoder_input_tokens.size(1))
        decoded = self.decoder(
            tgt=decoder_input,
            memory=history_memory,
            tgt_mask=causal_mask,
        )
        decoded = self.decoder_norm(decoded)
        return decoded

    def forward(self, history, target_red, target_blue):
        """训练前向：输出 6 个红球 logits 和 1 个蓝球 logits。"""
        history_memory, history_context = self.encode_history(history)
        decoder_input_tokens = self.build_decoder_inputs(target_red, target_blue)
        decoded = self.decode_from_encoded(history_memory, history_context, decoder_input_tokens)

        red_logits = self.red_head(decoded[:, :6, :])
        blue_logits = self.blue_head(decoded[:, 6, :])
        return red_logits, blue_logits

    def get_next_logits_from_encoded(self, history_memory, history_context, prefix_tokens):
        """
        采样时按 prefix 长度决定当前该预测红球还是蓝球。
        prefix 长度定义：
        - 1: [BOS]，预测第 1 个红球
        - 2..6: 继续预测红球
        - 7: 预测蓝球
        """
        decoded = self.decode_from_encoded(history_memory, history_context, prefix_tokens)
        step_index = prefix_tokens.size(1) - 1
        last_hidden = decoded[:, -1, :]

        if step_index < 6:
            return self.red_head(last_hidden), 'red'
        return self.blue_head(last_hidden), 'blue'


class LotAutoregressiveDataset(torch.utils.data.Dataset):
    """
    彩票滑窗数据集。

    每个样本：
    - history: [seq_length, 7]
    - target_red: [6]，标签范围 0..32
    - target_blue: []，标签范围 0..15
    """

    def __init__(self, data_path='./data/lot_data.csv', seq_length=72):
        super().__init__()
        self.data_path = data_path
        self.seq_length = seq_length

        data_df = pd.read_csv(self.data_path)
        columns = [f'red_ball_{ii}' for ii in range(6)] + ['blue_ball_0']
        self.data = data_df[columns].to_numpy(dtype=np.int64)
        self.dataset_length = len(self.data) - self.seq_length

        if self.dataset_length <= 0:
            raise ValueError(f'Not enough rows in {self.data_path} for seq_length={self.seq_length}')

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        history = torch.from_numpy(self.data[idx:idx + self.seq_length]).long()
        target = torch.from_numpy(self.data[idx + self.seq_length]).long()

        # CrossEntropyLoss 习惯使用从 0 开始的类别索引。
        target_red = target[:6] - 1
        target_blue = target[6] - 1
        return history, target_red, target_blue

    def get_latest_history(self):
        """采样时使用最近 72 期作为条件。"""
        return torch.from_numpy(self.data[-self.seq_length:]).long()


class DatasetSubset(torch.utils.data.Dataset):
    """简单子集包装器，避免 random_split。"""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]



def chronological_split(dataset, train_ratio=0.8, gap_size=None):
    """
    按时间顺序切分训练/验证集。

    规则：
    - 前 80% 窗口用于训练
    - 后 20% 窗口用于验证
    - 若数据量允许，中间留一个 seq_length 大小的 gap，减少窗口重叠泄漏
    """
    total_samples = len(dataset)
    if total_samples < 2:
        raise ValueError('Need at least 2 samples to create train/validation split.')

    train_end = max(1, int(total_samples * train_ratio))
    train_end = min(train_end, total_samples - 1)

    requested_gap = dataset.seq_length if gap_size is None else max(0, gap_size)
    max_gap = max(0, total_samples - train_end - 1)
    actual_gap = min(requested_gap, max_gap)

    val_start = train_end + actual_gap
    if val_start >= total_samples:
        actual_gap = 0
        val_start = train_end

    train_indices = range(0, train_end)
    val_indices = range(val_start, total_samples)

    if len(val_indices) == 0:
        raise ValueError('Validation split is empty after chronological split.')

    return DatasetSubset(dataset, train_indices), DatasetSubset(dataset, val_indices), actual_gap


class Config:
    """集中管理训练、验证、采样和模型超参数。"""

    def __init__(self):
        self.data_path = os.path.join(SCRIPT_DIR, 'data', 'lot_data.csv')
        self.model_path = os.path.join(SCRIPT_DIR, 'models', 'lot_autoregressive_classifier_scheme_a.pth')
        self.out_file = os.path.join(SCRIPT_DIR, 'lot_autoregressive_classifier_scheme_a.txt')
        self.device = DEVICE
        self.use_deterministic = False
        self.enable_tf32 = True
        self.use_amp = self.device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if self.device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        self.num_workers = min(4, os.cpu_count() or 1)
        self.pin_memory = self.device.type == 'cuda'
        self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = 4 if self.num_workers > 0 else None
        self.log_interval = 5
        self.enable_compile = False

        self.cond_seq_length = 72
        self.batch_size = 512 if self.device.type == 'cuda' else 128
        self.val_batch_size = 512 if self.device.type == 'cuda' else 128
        self.lr = 1e-4
        self.epochs = 30
        self.weight_decay = 1e-5
        self.gradient_clip = 1.0

        self.d_model = 256
        self.nhead = 8
        self.num_layers = 6
        self.dropout = 0.1

        self.sample_batch_size = 20
        self.temperature = 0.6
        self.top_k = 0
        self.train_ratio = 0.8
        self.split_gap = self.cond_seq_length

        configure_torch_runtime(
            self.device,
            deterministic=self.use_deterministic,
            enable_tf32=self.enable_tf32,
        )

        self.dataset = LotAutoregressiveDataset(
            data_path=self.data_path,
            seq_length=self.cond_seq_length,
        )
        self.train_dataset, self.val_dataset, self.actual_gap = chronological_split(
            self.dataset,
            train_ratio=self.train_ratio,
            gap_size=self.split_gap,
        )

        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'drop_last': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }
        val_loader_kwargs = {
            'batch_size': self.val_batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }
        if self.num_workers > 0:
            train_loader_kwargs['persistent_workers'] = self.persistent_workers
            train_loader_kwargs['prefetch_factor'] = self.prefetch_factor
            val_loader_kwargs['persistent_workers'] = self.persistent_workers
            val_loader_kwargs['prefetch_factor'] = self.prefetch_factor

        # 训练集可 shuffle；验证集不 shuffle。
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            **train_loader_kwargs,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            **val_loader_kwargs,
        )

        self.model = AutoregressiveLotteryModel(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            seq_length=self.cond_seq_length,
        ).to(self.device)
        if self.enable_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.red_criterion = nn.CrossEntropyLoss()
        self.blue_criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-5,
        )
        self.grad_scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp and self.amp_dtype == torch.float16)

    def autocast_context(self):
        if not self.use_amp:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)



def compute_loss(config, red_logits, blue_logits, target_red, target_blue):
    """总 loss = 6 个红球位置 CE + 1 个蓝球 CE 的平均值。"""
    flat_red_logits = red_logits.reshape(-1, red_logits.size(-1))
    flat_target_red = target_red.reshape(-1)
    red_loss = config.red_criterion(flat_red_logits, flat_target_red)
    blue_loss = config.blue_criterion(blue_logits, target_blue)
    return (red_loss * 6.0 + blue_loss) / 7.0


@torch.no_grad()
def validate(config):
    """在时间顺序验证集上评估平均 loss。"""
    config.model.eval()
    total_loss = 0.0
    total_samples = 0

    for history, target_red, target_blue in config.val_loader:
        history = history.to(config.device, non_blocking=config.pin_memory)
        target_red = target_red.to(config.device, non_blocking=config.pin_memory)
        target_blue = target_blue.to(config.device, non_blocking=config.pin_memory)

        with config.autocast_context():
            red_logits, blue_logits = config.model(history, target_red, target_blue)
            loss = compute_loss(config, red_logits, blue_logits, target_red, target_blue)

        batch_size = history.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else 0.0



def train():
    """训练入口：按 best val loss 保存 checkpoint。"""
    config = Config()
    model = config.model
    losses = []
    val_losses = []
    best_loss = float('inf')

    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    if os.path.exists(config.model_path):
        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                torch.load(config.model_path, map_location=config.device),
                strict=False,
            )
            model.train()
            if missing_keys or unexpected_keys:
                logging.info(
                    'Checkpoint loaded with non-strict matching. missing=%s unexpected=%s',
                    missing_keys,
                    unexpected_keys,
                )
            else:
                logging.info(f'Model loaded from {config.model_path}')
        except RuntimeError as e:
            logging.warning(f'Checkpoint at {config.model_path} is incompatible with current model structure: {e}')
            logging.warning('Training will start from randomly initialized Transformer weights.')

    logging.info(
        'Dataset split: total=%d, train=%d, val=%d, gap=%d, batch=%d, workers=%d, amp=%s, amp_dtype=%s',
        len(config.dataset),
        len(config.train_dataset),
        len(config.val_dataset),
        config.actual_gap,
        config.batch_size,
        config.num_workers,
        config.use_amp,
        str(config.amp_dtype).replace('torch.', ''),
    )

    for epoch_i in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(config.train_loader, desc=f'Epoch {epoch_i + 1}/{config.epochs}')

        for batch_i, (history, target_red, target_blue) in enumerate(progress_bar, start=1):
            history = history.to(config.device, non_blocking=config.pin_memory)
            target_red = target_red.to(config.device, non_blocking=config.pin_memory)
            target_blue = target_blue.to(config.device, non_blocking=config.pin_memory)

            config.optimizer.zero_grad(set_to_none=True)
            with config.autocast_context():
                red_logits, blue_logits = model(history, target_red, target_blue)
                loss = compute_loss(config, red_logits, blue_logits, target_red, target_blue)

            if config.grad_scaler.is_enabled():
                config.grad_scaler.scale(loss).backward()
                config.grad_scaler.unscale_(config.optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                config.grad_scaler.step(config.optimizer)
                config.grad_scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                config.optimizer.step()

            epoch_loss += loss.item()
            if batch_i % config.log_interval == 0 or batch_i == len(config.train_loader):
                progress_bar.set_postfix(loss=f'{loss.item():.4f}')

        config.lr_scheduler.step()
        val_loss = validate(config)

        avg_loss = epoch_loss / max(1, len(config.train_loader))
        losses.append(avg_loss)
        val_losses.append(val_loss)
        logging.info(
            'Epoch %d/%d, Train Loss: %.6f, Val Loss: %.6f',
            epoch_i + 1,
            config.epochs,
            avg_loss,
            val_loss,
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), config.model_path)
            logging.info(f'Best model saved to {config.model_path} with val loss {best_loss:.6f}')

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Scheme A Transformer)')
    plt.legend()
    plt.savefig(os.path.join(SCRIPT_DIR, 'lot_autoregressive_classifier_scheme_a_loss_curve.png'))
    plt.close()



def append_list_to_file(file_path, items_list):
    """沿用现有脚本风格：每行写一个 Python list。"""
    try:
        with open(file_path, 'a') as file:
            file.write(f'{items_list}\n')
    except IOError as e:
        print(f'An error occurred: {e}')



def sample_from_logits(logits, temperature=1.0, top_k=0):
    """
    从 logits 中采样。

    - temperature 控制随机性
    - top_k>0 时，只在前 k 个最高概率类别里采样
    """
    temperature = max(float(temperature), 1e-5)
    logits = logits / temperature

    if top_k is not None and top_k > 0 and top_k < logits.size(-1):
        top_values, _ = torch.topk(logits, k=top_k, dim=-1)
        threshold = top_values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def sample():
    """
    采样入口。

    生成顺序：
    1. 6 个红球
    2. 1 个蓝球

    对红球显式施加约束：
    - 已选红球不能重复
    - 必须严格递增
    - 必须保留足够大的剩余取值空间，以便后续位置还能继续递增生成
    """
    config = Config()
    model = config.model

    if os.path.exists(config.model_path):
        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                torch.load(config.model_path, map_location=config.device),
                strict=False,
            )
            model.eval()
            if missing_keys or unexpected_keys:
                logging.info(
                    'Checkpoint loaded with non-strict matching. missing=%s unexpected=%s',
                    missing_keys,
                    unexpected_keys,
                )
            else:
                logging.info(f'Model loaded from {config.model_path}')
        except RuntimeError as e:
            logging.warning(f'Checkpoint at {config.model_path} is incompatible with current model structure: {e}')
            logging.warning('Please rerun --train to create a Transformer checkpoint before sampling.')
            return

    history = config.dataset.get_latest_history().to(config.device)
    history = history.unsqueeze(0).expand(config.sample_batch_size, -1, -1)
    history_memory, history_context = model.encode_history(history)

    # 初始 prefix 只有 BOS。
    prefix_tokens = torch.full(
        (config.sample_batch_size, 1),
        model.bos_token_id,
        dtype=torch.long,
        device=config.device,
    )

    selected_reds = [[] for _ in range(config.sample_batch_size)]

    for red_position in range(6):
        red_logits, token_type = model.get_next_logits_from_encoded(history_memory, history_context, prefix_tokens)
        if token_type != 'red':
            raise RuntimeError('Expected red logits during red decoding step.')

        masked_logits = red_logits.clone()
        for batch_i in range(config.sample_batch_size):
            # 类别索引 0..32 对应实际号码 1..33。
            valid_mask = torch.ones(33, dtype=torch.bool, device=config.device)

            # 严格递增：当前红球必须 > 上一个红球。
            min_allowed = 1
            if selected_reds[batch_i]:
                min_allowed = selected_reds[batch_i][-1] + 1

            # 为剩余位置预留空间。
            # 例如当前位置是第 3 个红球，还剩 3 个红球要填，则当前最大只能到 30。
            remaining_after = 5 - red_position
            max_allowed = 33 - remaining_after

            valid_mask[:min_allowed - 1] = False
            valid_mask[max_allowed:] = False

            # 再次显式去掉已使用号码；虽然严格递增已隐含避免重复，但这里保留为显式约束。
            for used_value in selected_reds[batch_i]:
                valid_mask[used_value - 1] = False

            if not valid_mask.any():
                raise RuntimeError(f'No valid red numbers available at position {red_position}.')

            masked_logits[batch_i] = masked_logits[batch_i].masked_fill(~valid_mask, float('-inf'))

        # 红球 logits 的类别索引 0..32，采样后 +1 变为真实号码。
        next_red = sample_from_logits(masked_logits, temperature=config.temperature, top_k=config.top_k) + 1
        for batch_i in range(config.sample_batch_size):
            selected_reds[batch_i].append(int(next_red[batch_i].item()))

        # 红球在统一 token 词表里的 token id 与真实号码相同，直接拼接即可。
        prefix_tokens = torch.cat([prefix_tokens, next_red.unsqueeze(1)], dim=1)

    blue_logits, token_type = model.get_next_logits_from_encoded(history_memory, history_context, prefix_tokens)
    if token_type != 'blue':
        raise RuntimeError('Expected blue logits during blue decoding step.')

    # 蓝球 logits 类别索引 0..15，采样后 +1 变为真实蓝球号码。
    next_blue = sample_from_logits(blue_logits, temperature=config.temperature, top_k=config.top_k) + 1

    for batch_i in range(config.sample_batch_size):
        sample_numbers = selected_reds[batch_i] + [int(next_blue[batch_i].item())]
        print(f'{sample_numbers}')
        append_list_to_file(config.out_file, sample_numbers)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s[%(levelname)s][%(filename)s][%(lineno)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    args_parser = argparse.ArgumentParser(description='Train or sample from the autoregressive lottery model (Scheme A)')
    args_parser.add_argument('--train', action='store_true', help='Train the autoregressive classifier model')
    args_parser.add_argument('--sample', action='store_true', help='Sample from the autoregressive classifier model')
    args = args_parser.parse_args()

    if args.train:
        logging.info('Training the autoregressive classifier model...')
        train()
    elif args.sample:
        logging.info('Sampling from the autoregressive classifier model...')
        sample()
    else:
        logging.info('Please specify --train or --sample to run the script.')
        args_parser.print_help()
        exit(1)
