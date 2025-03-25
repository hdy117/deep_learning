import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
import os
import matplotlib.pyplot as plt
import json

# global file path
g_file_path=os.path.dirname(os.path.abspath(__file__))

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
learning_rate=0.01
n_epochs = 5
batch_size=512
img_size=28
input_size=28*28
out_dim=10
torch_model_path=os.path.join(g_file_path,".","model.pth")


# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为torch.Tensor类型，同时将像素值归一化到[0, 1]
    # transforms.Normalize((0.1307,), (0.3081,))  # 对数据进行归一化，这里的均值和标准差是MNIST数据集的统计值
])

# 加载训练集
train_dataset = torchvision.datasets.MNIST(root='./data',  # 数据存储的根目录
                                           train=True,  # 如果为True，则加载训练集
                                           download=True,  # 如果数据不存在，则自动下载
                                           transform=transform)

# 加载测试集
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,  # 如果为False，则加载测试集
                                          download=True,
                                          transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,  # 每个批次的样本数量
                                           shuffle=True)  # 是否在每个epoch打乱数据

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 位置编码
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

class ViT(nn.Module):
    def __init__(self, img_width, img_channels, patch_size, d_model, num_heads, num_layers, num_classes, ff_dim):
        super().__init__()

        self.patch_size = patch_size

        # given 7x7 flattened patch, map it into an embedding
        self.patch_embedding = nn.Linear(img_channels * patch_size * patch_size, d_model)

        # cls_token we are using we will be concatenating
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # (1, 4*4 + 1, 64)
        # + 1 because we add cls tokens
        self.position_embedding = nn.Parameter(
            torch.rand(1, (img_width // patch_size) * (img_width // patch_size) + 1, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # mapping 64 to 10 at the end
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        N, C, H, W = x.shape

        # we divide the image into 4 different 7x7 patches, and then flatten those patches
        # img shape will be 4*4 x 7*7
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(N, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, C * self.patch_size * self.patch_size)

        # each 7*7 flatten patch will be embedded to 64 dim vector
        x = self.patch_embedding(x)

        # cls tokens concatenated after repeating it for the batch
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        # learnable position embeddings added
        x = x + self.position_embedding

        # transformer takes 17x64 tensor, like it is a sequence with 17 words (17 because 4*4 + 1 from cls)
        x = self.transformer_encoder(x)

        # only taking the transformed output of the cls token
        x = x[:, 0]

        # mapping to number of classes
        x = self.fc(x)

        return x


# transformer parameter
embedding_dim=64
num_heads=4
num_layers=3

# model
vit=ViT(img_width=28,img_channels=1,patch_size=7, d_model=embedding_dim, num_heads=num_heads, num_layers=num_layers, num_classes=10, ff_dim=2048)
vit=vit.to(device)

# define train
def train():
    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(vit.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        n_total=0
        n_correct=0
        for idx, (samples, labels) in enumerate(train_loader):
            # print(f"len(samples):{len(samples)}, {samples.shape}")
            # print(f"len(labels):{len(labels)}, {labels.shape}")
            # to device
            samples=samples.to(device)
            labels=labels.to(device)

            # prediction
            y_pred = vit.forward(samples)

            # loss
            loss=criterion(y_pred,labels)

            # calculate gradients
            loss.backward()

            # gradient descent
            optimizer.step()
            optimizer.zero_grad()

            # accuracy
            if idx%10==0:
                _, predicted_indices = torch.max(y_pred, dim=1)
                n_total += y_pred.shape[0]
                n_correct += (predicted_indices == labels).sum().item()
                print(f'Epoch {epoch + 1}, Step {idx}, accuracy:{n_correct / n_total}, n_total:{n_total}, n_correct:{n_correct}')
        
        # save model
        torch.save(vit.state_dict(), torch_model_path)
                
def test():
    n_total=0
    n_correct=0

    # load model from file
    vit=ViT(img_width=28,img_channels=1,patch_size=7, d_model=embedding_dim, num_heads=num_heads, num_layers=num_layers, num_classes=10, ff_dim=2048)
    vit.load_state_dict(torch.load(torch_model_path))
    vit=vit.to(device)
    vit.eval()

    with torch.no_grad():
        for idx, (sample,label) in enumerate(test_loader):
            sample=sample.to(device)
            label=label.to(device)
            y_pred=vit.forward(sample)

            _, y_pred_idx=torch.max(y_pred,dim=1)
            n_correct+=(y_pred_idx==label).sum().item()
            n_total+=sample.shape[0]

        print(f'accuracy:{n_correct/n_total}, n_total:{n_total}, n_correct:{n_correct}')


if __name__ =="__main__":
    img, label=train_dataset[34]
    img=img.reshape(28,28)
    print(f'img.shape:{img.shape}, img type:{type(img)}, label.shape:{label}, label.type:{type(label)}')
    plt.imshow(img)
    plt.title(f'{label}')
    plt.show()

    train()
    test()
