import mnist_vae
from mnist_vae import *

# -------------------------
# Training and Testing
# -------------------------
epochs = 20

device=mnist_vae.device
model = mnist_vae.CVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# Loss Function
# -------------------------
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# -------------------------
# One-hot encoding for labels
# -------------------------
def one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes).float()

def train(epoch):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x = x.view(-1, 784).to(device)
        y_onehot = one_hot(y).to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x, y_onehot)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch}, Train Loss: {train_loss / len(train_loader.dataset):.4f}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(-1, 784).to(device)
            y_onehot = one_hot(y).to(device)
            recon_x, mu, logvar = model(x, y_onehot)
            test_loss += loss_function(recon_x, x, mu, logvar).item()
    print(f"Test Loss: {test_loss / len(test_loader.dataset):.4f}")

# -------------------------
# Training Loop
# -------------------------
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
