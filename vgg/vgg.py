import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import json

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
train_folder=os.path.join(g_file_path,"..","dataset","MNIST","mnist_train")
test_folder=os.path.join(g_file_path,"..","dataset","MNIST","mnist_test")
model_path=os.path.join(g_file_path,".",'model','vgg_7.onnx')

# define dateset
class MNISTDataset(Dataset):
    def __init__(self, dataset_path:str, transform=None):
        super().__init__()
        self.dataset_path=dataset_path
        self.transform=transform
        self.img_path=os.path.join(self.dataset_path,"samples")
        self.label_json_file_path=os.path.join(self.dataset_path,"labels.json")

        # list to save sample and label tensor
        self.img_list:list[torch.Tensor]=[]
        self.label:list[torch.Tensor]=[]

        # load add samples and labels
        with open(self.label_json_file_path, 'r') as label_handle:
            label_json=json.load(label_handle)

        # print(f'labels file path:{self.label_json_file_path}, label_json:{label_json}')
        
        # do loading
        for label in label_json:
            img_file=os.path.join(self.img_path, label['file'])
            img_read=Image.open(img_file)
            img_read=img_read.convert('L') # tor grayscale
            img_read=torch.from_numpy(np.array(img_read).astype(np.float32)).unsqueeze(0)
            self.img_list.append(img_read)

            label=label['label']
            label=torch.tensor(label)
            # print(f'label:{label.shape}')
            self.label.append(label)

    
    def __getitem__(self, index):
        img, label=self.img_list[index], self.label[index]
        if self.transform:
            img=self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_list)

# define dataloader
train_mnist_dataset=MNISTDataset(train_folder)
train_dataloader=DataLoader(dataset=train_mnist_dataset,batch_size=100,shuffle=True)

test_mnist_dataset=MNISTDataset(test_folder)
test_dataloader=DataLoader(dataset=test_mnist_dataset,batch_size=100,shuffle=True)

# define model
class VGGBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int,kernel_size:int=3):
        super().__init__()
        self.conv2d_1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=kernel_size,padding=kernel_size//2)
        self.relu=nn.LeakyReLU()
        self.conv2d_2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                              kernel_size=kernel_size,padding=kernel_size//2)
        self.max_pool=nn.MaxPool2d(2,2)
    
    def forward(self, x):
        out=self.conv2d_1(x)
        out=self.relu(out)
        out=self.conv2d_2(out)
        out=self.relu(out)
        out=self.max_pool(out)
        return out   

class VGG(nn.Module):
    def __init__(self,in_channel=1, out_channel=10):
        super().__init__()
        self.vgg_1=VGGBlock(in_channels=1, out_channels=8) # 14
        self.vgg_2=VGGBlock(in_channels=8, out_channels=16) # 7
        self.vgg_3=VGGBlock(in_channels=16, out_channels=32) # 3
        self.fc=nn.Linear(in_features=32*3*3, out_features=1024)
        self.relu=nn.LeakyReLU()
        self.fc_out=nn.Linear(in_features=1024, out_features=out_channel)
    
    def forward(self, x):
        out=self.vgg_1(x)
        out=self.vgg_2(out)
        out=self.vgg_3(out)
        out=out.view(-1, 32*3*3)
        out=self.fc(out)
        out=self.relu(out)
        out=self.fc_out(out)
        return out

# train
vgg_model=VGG(in_channel=1, out_channel=10)
vgg_model=vgg_model.to(device)

n_epoch=5
learning_rate=0.001
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(params=vgg_model.parameters(),lr=learning_rate)


def train():
    for epoch_idx in range(n_epoch):
        n_correct=0
        n_total=0
        for batch_idx, (sample, label) in enumerate(train_dataloader):
            # reset gradient
            optimizer.zero_grad()

            sample=sample.to(device)
            label=label.to(device)

            # predict
            y_pred=vgg_model(sample)

            # loss
            loss=criterion(y_pred, label)

            # calculate gradient
            loss.backward()

            # gradient descent
            optimizer.step()

            # test
            if batch_idx%10==0:
                _, y_pred_idx=torch.max(y_pred,1)
                n_correct+=(y_pred_idx==label).sum().item()
                n_total+=y_pred.shape[0]
                print(f'train accuracy:{n_correct/n_total}, n_correct:{n_correct}, n_total:{n_total}')

# test
def test():
    with torch.no_grad():
        n_correct=0
        n_total=0
        for idx, (sample, label) in enumerate(test_dataloader):
            sample=sample.to(device)
            label=label.to(device)

            # predict
            y_pred=vgg_model(sample)

            # test
            _, y_pred_idx=torch.max(y_pred,1)
            n_correct+=(y_pred_idx==label).sum().item()
            n_total+=y_pred.shape[0]
        print(f'test accuracy:{n_correct/n_total}, n_correct:{n_correct}, n_total:{n_total}')

# main
if __name__=="__main__":
    print(f'train data path:{train_folder}')
    print(f'test data path:{test_folder}')
    train()

    # save model
    torch.onnx.export(
        vgg_model,  # PyTorch model to be exported
        torch.randn(size=[1, 28, 28],dtype=torch.float32).to(device),  # Sample input tensor
        model_path,  # Path to save the ONNX model
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=12,  # ONNX version to use
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=['input'],  # Names to assign to the input tensors
        output_names=['output'],  # Names to assign to the output tensors
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Specify dynamic axes
    )   

    test()
    