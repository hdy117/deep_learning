import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

batch_size=128

transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test=transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_dataset=torchvision.datasets.CIFAR10(root='../dataset/', train=True, download=True, transform=transform_train)
test_dataset=torchvision.datasets.CIFAR10(root='../dataset/', train=False, download=True, transform=transform_test)

train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if __name__=="__main__":
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # 反标准化以显示原始图像
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    images = images * std + mean  # 还原至 [0, 1] 范围
    images = images.numpy().transpose((0, 2, 3, 1))  # 转换为 (batch, H, W, C) 格式
    
    print(f'images[0]:{images[0]}')
    print(f'labels[0]:{labels[0]}')

    # 绘制图像
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(images[i])
        plt.title(train_dataset.classes[labels[i]])
        plt.axis('off')
    plt.show()