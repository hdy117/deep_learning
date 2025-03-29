import torch
import torch.nn as nn


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def apply_rotary_pos_emb(self, x, cos, sin):
        x_embed = (x * cos) + (rotate_half(x) * sin)
        return x_embed

    def forward(self, q, k):
        seq_len=q.shape[-2] # get seq length
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return self.apply_rotary_pos_emb(q, cos, sin), self.apply_rotary_pos_emb(k, cos, sin)


if __name__ == "__main__":
    # 示例使用
    dim = 64
    seq_len = 10
    rotary_encoding = RotaryPositionalEncoding(dim)

    # 假设 q 和 k 是查询和键张量
    q = torch.randn(256, seq_len, dim)
    k = torch.randn(256, seq_len, dim)

    q_embed, k_embed= rotary_encoding(q, k)
    print("Encoded query shape:", q_embed.shape)
    print("Encoded key shape:", k_embed.shape)