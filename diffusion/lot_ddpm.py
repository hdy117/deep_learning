import torch
import torch.nn as nn
import torch.nn.functional as  F
import math, os, tqdm
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import logging

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        
        # Compute positional encoding
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sine
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cosine
        
        # Register the encoding as a buffer (move it to the correct device when the model is moved)
        self.register_buffer('encoding', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # self.encoding shape: [1, max_len, d_model]
        # Add positional encoding to x (slice to match seq_len)
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class ConditionalEncoder(nn.Module):
    def __init__(self, input_dim=7, model_dim = 128, num_layers=1, nhead=8):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, model_dim),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True,dim_feedforward=model_dim*4, activation='gelu'), 
            num_layers=num_layers
        )
        self.positonal_encoding=PositionalEncoding(model_dim, max_len=model_dim+1)  # Positional encoding for transformer
        self.class_token=nn.Parameter(torch.randn(1,1,model_dim)) # Class token for transformer

    def forward(self, x):
        x = self.embedding(x)  #(batch, sequence, input_dim -> model_dim)
        
        # add class token
        class_token=self.class_token.expand(x.size(0),-1,-1)   #(batch, 1, model_dim)
        x = torch.cat([class_token, x], dim=1)
        
        # positional encoding
        x = self.positonal_encoding(x)
        x = self.encoder(x)    #(batch, sequence, model_dim)
        
        return x[:,0]        #(batch, model_dim) as condition embedding, using the class token output

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [batch]
        return: [batch, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class LinearBlock(nn.Module):
    def __init__(self, im_dim, out_dim, embedding_dim=128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(im_dim, out_dim), # (batch_size, im_dim) -> (batch_size, out_dim)
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),  # (batch_size, out_dim) -> (batch_size, out_dim)
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),  # (batch_size, out_dim) -> (batch_size, out_dim)
        )
        self.embedding_proj = nn.Linear(embedding_dim, im_dim)  # embedding projection to match output channels
        self.short_cut=nn.Linear(im_dim, out_dim) if im_dim != out_dim else nn.Identity()  # shortcut connection to match dimensions
        
        self.out_activation=nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self,x,embedding): 
        # x:(batch_size, im_dim, sequence)
        # embedding: (batch_size, embedding_dim)
        embedding_proj=self.embedding_proj(embedding)  # (batch_size, out_dim)
        x=x+embedding_proj  # add embedding projection to input
        
        x_short=self.short_cut(x)  # shortcut connection
        
        x=self.block(x)  # (batch_size, out_dim)
        
        x=x+x_short  # add shortcut connection

        return self.out_activation(x) # (batch_size, out_dim)
    
class UNet1D(nn.Module):
    def __init__(self, in_dim=2, base_dim=32, embedding_dim=128, num_cond_feature=7):
        super().__init__()
        
        # step t embedding, [batch_size] --> [batch_size, t_embed_dim]
        self.t_embedding=nn.Sequential(
            SinusoidalPositionEmbeddings(embedding_dim),
            nn.Linear(embedding_dim, 4*embedding_dim),
            nn.GELU(),
            nn.Linear(4*embedding_dim, embedding_dim),
        )
        
        # condition embedding, [batch_size, sequence, cond_dim] --> [batch_size, cond_dim]
        self.condition_embedding = ConditionalEncoder(input_dim=num_cond_feature, model_dim=embedding_dim) 
    
        # inital projection
        self.init_proj=nn.Sequential(
            nn.Linear(in_dim, base_dim),  # embedding projection to match output channels
            nn.GELU(),
        )
        
        # down blocks 
        self.down1 = LinearBlock(base_dim, base_dim * 2, embedding_dim=embedding_dim)  #(batch_size, basechannles, l) -> (batch_size, basechannels * 2, l)
        self.down2 = LinearBlock(base_dim * 2, base_dim * 4, embedding_dim=embedding_dim)  # -> (batch_size, basechannels * 4, l)

        # bottleneck
        self.bottleneck = LinearBlock(base_dim * 4, base_dim * 4, embedding_dim=embedding_dim)
        
        # up blocks
        self.up1 = LinearBlock(base_dim * 8, base_dim * 2, embedding_dim=embedding_dim)
        self.up2 = LinearBlock(base_dim * 4, base_dim * 1, embedding_dim=embedding_dim)

        # output projection
        self.out_proj = nn.Sequential(
            nn.Linear(base_dim, in_dim), 
        )

    def forward(self, x:torch.Tensor, t:torch.Tensor, cond:torch.Tensor):
        # x : [batch_size, out_dim]
        # t : [batch_size]
        # cond : [batch_size, sequence, conditon_dim]
        
        # init conv
        x=self.init_proj(x)  # (batch_size, out_dim) -> (batch_size, base_dim)
        
        # step t embedding
        t_embedding=self.t_embedding(t)  # (batch_size, t_embed_dim)
        
        # condition embedding
        cond_embedding = self.condition_embedding(cond)  # (batch_size, cond_dim)
        
        # embeddings
        embeddings=t_embedding + cond_embedding  # (batch_size, embed_dim)

        # dowsnsample
        x1 = self.down1(x,embeddings)  # (batch_sizeb, base_dim) -> (batch_size, base_dim*2)
        x2 = self.down2(x1,embeddings)  # (batch_size, base_dim*2) -> (batch_size, base_dim*4)
        
        # bottleneck
        x3 = self.bottleneck(x2,embeddings)  # (batch_size, base_dim*4)

        # upsample
        x = self.up1(torch.cat([x3, x2], dim=1),embeddings) # (batch_size, base_dim*8) -> (batch_size, base_dim*2)
        x = self.up2(torch.cat([x, x1], dim=1),embeddings) # (batch_size, base_dim*4) -> (batch_size, base_dim)

        return self.out_proj(x)   # (batch_size, base_dim) -> [batch_size,out_dim]
    
class Diffusion_model(nn.Module):
    def __init__(self, timestep=1000, num_cond_feature=7, num_out_dim=7):
        super().__init__()
        
        self.num_cond_feature=num_cond_feature
        self.num_out_dim=num_out_dim
        self.unet = UNet1D(in_dim=self.num_out_dim, base_dim=32, embedding_dim=128, num_cond_feature=self.num_cond_feature) # 

        self.time_steps = timestep        
        betas = torch.linspace(1e-4, 0.02, timestep)
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

    def forward_diffusion(self,x0,t,noise):
        # add noise to x0 to get x_t
        sqrt_alpha=self.alphas_cumprod[t].view(-1,1) # (batch_size, 1)
        sqrt_one_minus=self.sqrt_one_minus_alphas_cumprod[t].view(-1,1)
        x_t = sqrt_alpha*x0 + sqrt_one_minus*noise
        
        return x_t

    def p_loss(self, x0:torch.Tensor, condition:torch.Tensor, cond_drop_ratio=0.2):
        # condition: [batch_size, sequence, cond_dim]
        # x0: [batch_size, out_dim]
        
        # batch size
        batch_size = condition.shape[0]

        # random step t
        t = torch.randint(0, self.time_steps, (batch_size,), device=condition.device)
        
        # generate noise
        noise = torch.randn_like(x0)
        
        # add noise
        x_t=self.forward_diffusion(x0, t, noise)
        
        # random drop condition
        keep_mask = (torch.rand(batch_size, device=condition.device) > cond_drop_ratio).float()[:, None, None]  # (batch_size, 1, 1)
        condition = condition * keep_mask  # randomly drop some conditions

        # predict noise using unet
        pred_noise = self.unet(x_t, t, condition)
        
        return pred_noise, noise
    
    @torch.no_grad()
    def forward(self, condition:torch.Tensor)->torch.Tensor:
        # condition: [batch_size, sequence, cond_dim]
        # return:[batch_size, out_dim]
        steps = self.time_steps
        batch_size = condition.shape[0]
        
        guidance_scale=3.0
        
        x = torch.randn(batch_size, self.num_out_dim).to(condition.device) # [batch_size, out_dim]

        batch_size=batch_size*2  # for guidance scale, we double the batch size
        condition_concat=torch.concat([condition, torch.zeros_like(condition)],dim=0) # condition and unconditioned
        
        for i in reversed(range(steps)):
            t = torch.full((batch_size,), i, device=condition.device, dtype=torch.long) # (batch_size, )
            
            x_concat=torch.concat([x, x],dim=0)  # duplicate x for guidance scale
            noise:torch.Tensor=self.unet(x_concat, t, condition_concat)
            noise_chunks = torch.chunk(noise, chunks=2,dim=0)  # split the noise for conditioned and unconditioned
            pred_noise = noise_chunks[0]
            pred_noise_uncond = noise_chunks[1]
            if guidance_scale > 1.0:
                pred_noise = pred_noise + guidance_scale * (pred_noise - pred_noise_uncond)
            
            # pred_noise = self.unet(x, t, condition)
            
            # if guidance_scale > 1.0:
            #     pred_noise_uncond=self.unet(x,t,torch.zeros_like(condition))  # unconditioned prediction
            #     pred_noise = pred_noise + guidance_scale * (pred_noise - pred_noise_uncond)
            
            t=torch.chunk(t, chunks=2,dim=0)[0]  # use the first chunk for time steps
            beta_t=self.betas[t].view(-1, 1)
            alpha_t=self.alphas[t].view(-1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1)
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod_t) # (batch_size, out_dim)
            
            if i > 0:
                noise = torch.randn_like(x)
                var = self.posterior_variance[t].view(-1, 1)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean

        return x 
    
# dataset
class LotDataset(torch.utils.data.Dataset):
    def __init__(self, data_path=f'./data/lot_data.csv', seq_length=72, out_dim=7, pre_scale=16.5, post_scale=8.0):
        super().__init__()
        self.data_path = data_path
        self.data=pd.read_csv(self.data_path)  # load data from csv
        self.seq_length=seq_length
        self.out_dim=out_dim
        self.pre_scale=pre_scale
        self.post_scale=post_scale
        
        # length of dataset
        self.dataset_length = len(self.data)-self.seq_length-1

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        condition=torch.zeros((self.seq_length+1, self.out_dim))  # condition input, [seq_length+1, condition_feature_dim]
        for i in range(self.seq_length+1):
            item_list=[]
            for ii in range(self.out_dim-1):
                item=int(self.data[f'red_ball_{ii}'].iloc[idx+i])  # extract red info from data
                item_list.append(item)
            item_list.append(int(self.data[f'blue_ball_0'].iloc[idx+i]))  # extract blue info from data
            condition[i]=torch.tensor(item_list, dtype=torch.float)
            
        # scale condition
        pre_cond=(condition[:,0:self.out_dim-1]-self.pre_scale)/self.pre_scale
        post_cond=(condition[:,(self.out_dim-1):]-self.post_scale)/self.post_scale
        condition_scale=torch.cat((pre_cond, post_cond), dim=1)  # condition, [seq_length+1, out_dim]
        
        # extract condition
        condition_scale=condition_scale[0:self.seq_length]  # condition, [seq_length, out_dim]
        
        # extract x0
        x0=condition_scale[-1]  # last item in condition, [out_dim]
        
        condition_scale = condition_scale.to(torch.float) # ensure condition is in int format
        x0=x0.to(torch.float)  # ensure x0 is in float format
                
        return condition_scale, x0  # ensure the data is in float format  

# config
class Config:
    def __init__(self):
        self.data_path='./data/lot_data.csv'  # path to the dataset
        self.model_path='./models/lot_ddpm.pth'  # path to save the model
        
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device to use
        
        self.pre_scale=16.5  # pre scale for the first 6 items
        self.post_scale=8.0  # post scale for the last item
        
        self.cond_seq_lenth=72  # condition sequence length
        self.condition_feature_dim=7  # condition input dim
        self.out_dim=7 # output dimension
        self.ddpm_scheduler_steps=1000  # number of diffusion steps
        self.ddpm_model=Diffusion_model(timestep=self.ddpm_scheduler_steps, num_cond_feature=self.condition_feature_dim, num_out_dim=self.out_dim).to(self.device)
        
        # dataset and dataloader
        self.dataset=LotDataset(data_path=self.data_path, seq_length=self.cond_seq_lenth, out_dim=self.out_dim,pre_scale=self.pre_scale, post_scale=self.post_scale)
        self.data_loader=torch.utils.data.DataLoader(dataset=self.dataset, batch_size=128, shuffle=True)
        
        self.lr=1e-4
        self.epochs=500
        self.optimizer=torch.optim.Adam(self.ddpm_model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.criterion=nn.MSELoss()
        self.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=10,T_mult=1,eta_min=1e-5)
        
        self.sample_batch_size=25

# training loop
def train():
    config:Config = Config()
    ddpm_model:Diffusion_model=config.ddpm_model
    losses = []
    
    # load if model exists
    if os.path.exists(config.model_path):
        ddpm_model.load_state_dict(torch.load(config.model_path))
        ddpm_model.train()
        logging.info(f'Diffusion model loaded from {config.model_path}')
    
    for epoch_i in range(config.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(config.data_loader, desc=f"Epoch {epoch_i+1}/{config.epochs}")
        
        for one_batch in progress_bar:
            # clear gradients
            config.optimizer.zero_grad()
            
            # get condition and x0
            condition=one_batch[0].to(config.device)  # [batch_size, sequence, condition_feature_dim]
            x0=one_batch[1].to(config.device)  # [batch_size, out_dim]
            
            logging.debug(f'condition.shape: {condition.shape}, x0.shape: {x0.shape}')
            
            # forward diffusion
            pred_noise, noise = ddpm_model.p_loss(x0, condition)
            
            # loss
            loss=config.criterion(pred_noise, noise)
            loss.backward()
            config.optimizer.step()
            
            # update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # update learning rate
        config.lr_scheduler.step()
        
        # save model
        if epoch_i % 10 == 0:
            torch.save(ddpm_model.state_dict(), config.model_path)
            logging.info(f'Model saved to {config.model_path}')
        
        avg_loss = epoch_loss / len(config.data_loader)
        losses.append(avg_loss)
        logging.info(f"Epoch {epoch_i+1}/{config.epochs}, Average Loss: {avg_loss:.6f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('./lot_ddpm_loss_curve.png')
    plt.close()

# sample
def sample():
    config:Config = Config()
    ddmp_model:Diffusion_model=config.ddpm_model
    
    if os.path.exists(config.model_path):
        ddmp_model.load_state_dict(torch.load(config.model_path))
        ddmp_model.eval()
        logging.info(f'Diffusion model loaded from {config.model_path}')
    else:
        logging.info(f'No model found at {config.model_path}, please train the model first.')
        return

    with torch.no_grad():
        ddmp_model.to(config.device)
        
        # load the last condition from the dataset
        condition, _=config.dataset[-1]
        condition = condition.to(config.device)
        
        # expand condition to match the sample batch size
        condition=torch.expand(condition, (config.sample_batch_size, -1, -1)) # expand condition to [sample_batch_size, sequence, condition_feature_dim]
        
        # sample from the model
        samples=ddmp_model.forward(condition)
        
        # output transform
        samples=samples.cpu()
        for batch_i in range(samples.shape[0]):
            sample=samples[batch_i]
            sample=sample.view(config.out_dim) # reshape to [out_dim]
            pre_sample=(sample[0:(config.out_dim-1)]+1.0)*config.pre_scale
            post_sample=(sample[(config.out_dim-1):]+1.0)*config.post_scale
            pre_sample=torch.clip(pre_sample.astype(int),1,int(2*config.pre_scale))
            post_sample=torch.clip(post_sample.astype(int),1,int(2*config.post_scale))
            print('{pre_sample}, {post_sample}')
            
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    args_parser=argparse.ArgumentParser(description="Train or sample from the Lot DDPM model")
    args_parser.add_argument('--train', action='store_true', help='Train the DDPM model')
    args_parser.add_argument('--sample', action='store_true', help='Sample from the DDPM Model')
    args = args_parser.parse_args()
    
    if args.train:
        logging.info(f'Training the DDPM model...')
        train()
    elif args.sample:  
        logging.info(f'Sampling from the DDPM model...')     
        sample()
    else:
        logging.info("Please specify --train or --sample to run the script.")
        args_parser.print_help()
        exit(1)          
    

