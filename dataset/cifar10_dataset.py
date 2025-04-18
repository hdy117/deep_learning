import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

g_file_path=os.path.dirname(os.path.abspath(__file__))
Image_Size=224

class CustomCIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.images = []
        self.labels = []

        if self.train:
            for i in range(1, 6):
                batch_path = os.path.join(data_dir, f'data_batch_{i}')
                with open(batch_path, 'rb') as f:
                    batch_data = pickle.load(f, encoding='bytes')
                self.images.append(batch_data[b'data'])
                self.labels.extend(batch_data[b'labels'])
            self.images = np.concatenate(self.images, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        else:
            test_batch_path = os.path.join(data_dir, 'test_batch')
            with open(test_batch_path, 'rb') as f:
                test_data = pickle.load(f, encoding='bytes')
            self.images = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.labels = test_data[b'labels']
        
        print(f'{Counter(self.labels)}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # to PIL Image
        image=Image.fromarray(image)

        # 将标签转换为one - hot编码
        one_hot_label = torch.zeros(10)
        one_hot_label[label] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, one_hot_label

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((Image_Size,Image_Size),interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
])

transform_no_resize = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

data_dir = os.path.join(g_file_path,'OpenDataLab___CIFAR-10','raw','cifar-10-python','cifar-10-batches-py')

def show_img(image, label):
    # 反归一化操作，恢复图像原始像素值范围
    # mean = np.array([0.4914, 0.4822, 0.4465])
    # std = np.array([0.2023, 0.1994, 0.2010])
    image = image.numpy().transpose(1, 2, 0)  # 将通道维度移到最后
    print(f'{image}')
    # image = image * std + mean
    image = np.clip(image, 0, 1)  # 确保像素值在[0, 1]之间
    print(f'image.shape:{image.shape}')

    # 显示图像
    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.show()


if __name__=="__main__":
    # 创建训练集
    train_dataset = CustomCIFAR10Dataset(data_dir, train=True, transform=transform)
    test_dataset = CustomCIFAR10Dataset(data_dir, train=False, transform=transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    index = 12
    # 从数据集中获取一张图像和对应的标签
    image, label = train_dataset[index]  # 这里取索引为0的图像，你可以修改索引值
    show_img(image,label)

    image, label = test_dataset[index]  # 这里取索引为0的图像，你可以修改索引值
    show_img(image,label)