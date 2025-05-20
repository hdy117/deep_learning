import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Hyperparameters
# -------------------------
batch_size = 128
latent_dim = 2
hidden_dim = 400
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./cvae_mnist.pth"

# -------------------------
# Data
# -------------------------
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# -------------------------
# Conditional VAE
# -------------------------
class CVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2, num_classes=10):
        super(CVAE, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, y):
        x = torch.cat([x, y], dim=1)
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        z = torch.cat([z, y], dim=1)
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar



