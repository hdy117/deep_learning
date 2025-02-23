import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
n_epochs = 3
n_batch_size=16
cnn_in_channel=1
out_dim=10
model_path=os.path.join(g_file_path,".","model","model_cnn.onnx")

# prepare dataset
class MNISTDataset(Dataset):
    def __init__(self, dataset_path:str, transform=None):
        super().__init__()

        self.transform=transform

        # path
        self.dataset_path=dataset_path
        self.sample_path=os.path.join(self.dataset_path,'samples')
        self.label_path=os.path.join(self.dataset_path,'labels.json')

        # load labels.json
        label_json_list=[]
        with open(self.label_path,'r') as label_handle:
            label_json_list=json.load(label_handle)

        # samples and labels
        self.samples:list[np.array]=[]
        self.labels:list[torch.Tensor]=[]

        # load samples 
        for label_json in label_json_list:
            # image tensor
            img_path=os.path.join(self.sample_path,label_json['file'])
            img_array:np.array=np.array(Image.open(img_path).convert('L'))
            self.samples.append(img_array)
            
            # label tensor
            label=label_json['label']
            label_tensor=torch.zeros(10)
            label_tensor[label]=1
            self.labels.append(torch.tensor(label))
                
    def __getitem__(self, index):
        sample, label=self.samples[index], self.labels[index]
        if self.transform:
            sample=self.transform(sample)
        return sample, label
    
    def __len__(self):
        return len(self.labels) 

# prepare dataloader
train_dataset_path=os.path.join(g_file_path,"..","dataset","mnist","mnist_train")
mnist_train_dataset=MNISTDataset(dataset_path=train_dataset_path, transform=transforms.ToTensor())
train_dataloader=DataLoader(dataset=mnist_train_dataset, shuffle=True, batch_size=n_batch_size)

test_data_path=os.path.join(g_file_path,"..","dataset","mnist","mnist_test")
mnist_test_dataset=MNISTDataset(dataset_path=test_data_path, transform=transforms.ToTensor())
test_dataloader=DataLoader(dataset=mnist_test_dataset, shuffle=True, batch_size=n_batch_size)

# define model
class MnistModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim=in_dim
        self.conv2_1=nn.Conv2d(in_channels=in_dim, out_channels=4,kernel_size=5) 
        self.max_pool=nn.MaxPool2d(2,2)
        self.conv2_2=nn.Conv2d(in_channels=4, out_channels=8,kernel_size=5) 
        self.linear_1=nn.Linear(in_features=8*4*4, out_features=32)
        self.linear_2=nn.Linear(in_features=32, out_features=out_dim)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        # print(f'x.shape:{x.shape}')
        # x=x.view(-1, self.in_dim)
        out=self.conv2_1(x) # (28-5+2*0)/1+1 --> 24
        out=self.max_pool(out) # 24/2 --> 12
        out=self.conv2_2(out) # (12-5+2*0)/1+1 --> 8
        out=self.max_pool(out) # 8/2 --> 4
        out=out.view(-1, 8*4*4)
        out=self.linear_1(out)
        out=self.relu(out)
        out=self.linear_2(out)
        return out

# model
mnist_model=MnistModel(in_dim=cnn_in_channel, out_dim=out_dim)
mnist_model=mnist_model.to(device)

# define train
def train():
    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(mnist_model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        n_total=0
        n_correct=0
        for idx, (samples, labels) in enumerate(train_dataloader):
            # print(f"len(samples):{len(samples)}, {samples.shape}")
            # print(f"len(labels):{len(labels)}, {labels.shape}")
            # to device
            samples=samples.to(device)
            labels=labels.to(device)

            # prediction
            y_pred = mnist_model.forward(samples)

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
                
def test():
    n_total=0
    n_correct=0
    with torch.no_grad():
        for idx, (sample,label) in enumerate(test_dataloader):
            sample=sample.to(device)
            label=label.to(device)
            y_pred=mnist_model.forward(sample)

            _, y_pred_idx=torch.max(y_pred,dim=1)
            n_correct+=(y_pred_idx==label).sum().item()
            n_total+=sample.shape[0]

        print(f'accuracy:{n_correct/n_total}, n_total:{n_total}, n_correct:{n_correct}')


if __name__ =="__main__":
    img, label=mnist_train_dataset[34]
    img=img.reshape(28,28)
    print(f'img.shape:{img.shape}, img type:{type(img)}, label.shape:{label.shape}, label.type:{type(label)}')
    plt.imshow(img)
    plt.title(f'{label}')
    plt.show()

    train()

    # save model
    torch.onnx.export(
        mnist_model,  # PyTorch model to be exported
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
