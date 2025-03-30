import torch
import torch.nn as nn
import RoPE


class TransEncoder(nn.Module):
    def __init__(self, feat_dim:int=64, d_model:int=768, num_heads:int=4, num_layers:int=3):
        super().__init__()
        self.feat_dim=feat_dim
        self.d_model=d_model
        self.num_heads=num_heads
        self.num_layers=num_layers
        
        self.embeding=nn.Linear(self.feat_dim, self.d_model)
        self.RoPE_encoding=RoPE.RotaryPositionalEncoding(dim=self.d_model)
        self.trans_encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model,nhead=self.num_heads),
            num_layers=self.num_layers
        )
        
    def forward(self,x):
        # input/output x is [batch, seq, feat]
        x=self.embeding(x)
        x=self.RoPE_encoding(x)
        x=x.permute(1,0,2) # [seq, batch, feat]
        x=self.trans_encoder(x)
        x=x.permute(1,0,2) # [batch, seq, feat]
        return x

