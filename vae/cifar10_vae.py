import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# --- Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_data = datasets.CIFAR10(root='../dataset', train=True, download=True, transform=transform)
loader = DataLoader(train_data, batch_size=128, shuffle=True)

# --- VAE Model ---
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 32x16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 64x8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 128x4x4
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128*4*4, latent_dim)
        self.fc_logvar = nn.Linear(128*4*4, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 64x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 32x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), # 3x32x32
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 128, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar, z

# --- Loss ---
def vae_loss(x_recon, x, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

# --- Training ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Training VAE...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for x, _ in loader:
        x = x.to(device)
        x_recon, mu, logvar, _ = model(x)
        loss = vae_loss(x_recon, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader.dataset):.4f}")

# --- Visualization: 2D t-SNE latent space ---
print("Extracting latent vectors for t-SNE...")
model.eval()
latents = []
labels = []
with torch.no_grad():
    for x, y in DataLoader(train_data, batch_size=256):
        x = x.to(device)
        _, _, _, z = model(x)
        latents.append(z.cpu())
        labels.append(y)
        if len(latents) > 20: break  # for speed

latents = torch.cat(latents).numpy()
labels = torch.cat(labels).numpy()

print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
z_tsne = tsne.fit_transform(latents)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(z_tsne[:,0], z_tsne[:,1], c=labels, cmap='tab10', s=10)
plt.colorbar(scatter, ticks=range(10))
plt.title("VAE Latent Space t-SNE (CIFAR-10)")
plt.show()

# --- Interpolation ---
def interpolate(z1, z2, steps=10):
    return [(1 - alpha) * z1 + alpha * z2 for alpha in torch.linspace(0, 1, steps)]

# 从测试集中选两张不同类图像
x1 = train_data[3][0].unsqueeze(0).to(device)
x2 = train_data[8][0].unsqueeze(0).to(device)

with torch.no_grad():
    mu1, _ = model.encode(x1)
    mu2, _ = model.encode(x2)
    zs = interpolate(mu1[0], mu2[0])
    recon = [model.decode(z.unsqueeze(0)).squeeze().cpu() for z in zs]

plt.figure(figsize=(15, 2))
for i, img in enumerate(recon):
    plt.subplot(1, len(recon), i+1)
    plt.imshow(img.permute(1,2,0))
    plt.axis('off')
plt.suptitle("Latent Interpolation between Two CIFAR-10 Images")
plt.show()
