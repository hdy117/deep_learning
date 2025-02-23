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
n_epochs = 5
n_batch_size=16
logistic_in_dim=28*28
out_dim=10
model_path=os.path.join(g_file_path,".","model","model_logistic.onnx")

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
            self.labels.append(label_tensor)
                
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
        self.hidden_size_1=256
        self.linear_1=nn.Linear(in_features=in_dim, out_features=self.hidden_size_1)
        self.relu=nn.ReLU()
        self.linear_2=nn.Linear(in_features=self.hidden_size_1, out_features=out_dim)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,x):
        x=x.view(-1, self.in_dim)
        out=self.linear_1(x)
        out=self.relu(out)
        out=self.linear_2(out)
        out=self.sigmoid(out)
        return out

# model
mnist_model=MnistModel(in_dim=logistic_in_dim, out_dim=out_dim)
mnist_model=mnist_model.to(device)

# define train
def train():
    # criterion
    criterion = nn.BCELoss()

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
                _, labels_indices = torch.max(labels, dim=1)
                n_total += y_pred.shape[0]
                n_correct += (predicted_indices == labels_indices).sum().item()
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
            _, label_idx=torch.max(label,dim=1)
            n_correct+=(y_pred_idx==label_idx).sum().item()
            n_total+=sample.shape[0]

        print(f'accuracy:{n_correct/n_total}, n_total:{n_total}, n_correct:{n_correct}')


if __name__ =="__main__":
    img, label=mnist_train_dataset[34]
    img=img.reshape(28,28)
    print(f'img.shape:{img.shape}, img type:{type(img)}, img:{img}, label.shape:{label.shape}, label.type:{type(label)}')
    plt.imshow(img)
    plt.title(f'{label}')
    plt.show()

    train()

    # save model
    torch.onnx.export(
        mnist_model,  # PyTorch model to be exported
        torch.randn(size=[1, 28*28],dtype=torch.float32).to(device),  # Sample input tensor
        model_path,  # Path to save the ONNX model
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=12,  # ONNX version to use
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=['input'],  # Names to assign to the input tensors
        output_names=['output'],  # Names to assign to the output tensors
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Specify dynamic axes
    )

    test()
