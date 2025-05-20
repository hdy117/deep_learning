import mnist_vae
from mnist_vae import *

model= mnist_vae.CVAE()
model.load_state_dict(torch.load(mnist_vae.model_path))
model=model.to(mnist_vae.device)

# -------------------------
# Latent Space Visualization
# -------------------------
def plot_latent_space(model, data_loader):
    model.eval()
    zs, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.view(-1, 784).to(device)
            y_onehot = one_hot(y).to(device)
            mu, _ = model.encode(x, y_onehot)
            zs.append(mu.cpu().numpy())
            labels.append(y.numpy())
    zs = np.concatenate(zs)
    labels = np.concatenate(labels)
    plt.figure(figsize=(8, 6))
    plt.scatter(zs[:, 0], zs[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar()
    plt.title("Latent Space of CVAE")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.grid(True)
    plt.show()

plot_latent_space(model, test_loader)

# -------------------------
# Generation From Latent Space
# -------------------------
def generate_images(model, num_classes=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_classes, latent_dim).to(device)
        labels = torch.arange(num_classes)
        y_onehot = one_hot(labels).to(device)
        samples = model.decode(z, y_onehot).cpu()
        samples = samples.view(-1, 1, 28, 28)
        grid = torch.cat([samples[i] for i in range(num_classes)], dim=2)
        plt.imshow(grid.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title("Generated Digits by Class")
        plt.show()

generate_images(model)