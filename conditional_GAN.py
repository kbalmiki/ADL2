import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tqdm
import os
import wandb

wandb.login(key="c224d69435c53c15d36dfd73732ff7a3ee70c463")
wandb.init(project="conditional-gan-mnist")

# Hyperparameters
mb_size = 64
Z_dim = 100
h_dim = 128
lr = 1e-3
num_classes = 10

# Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root='../MNIST', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=mb_size, shuffle=True)

X_dim = 784

# Xavier Initialization
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Generator with label conditioning
class Generator(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim, num_classes):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.fc1 = nn.Linear(z_dim + num_classes, h_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.apply(xavier_init)

    def forward(self, z, labels):
        c = self.label_embedding(labels)
        x = torch.cat([z, c], dim=1)
        h = F.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out

# Discriminator with label conditioning
class Discriminator(nn.Module):
    def __init__(self, x_dim, h_dim, num_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.fc1 = nn.Linear(x_dim + num_classes, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)
        self.apply(xavier_init)

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out

# Training loop for CGAN
def cGANTraining(G, D, loss_fn, train_loader):
    G.train()
    D.train()

    D_loss_real_total = 0
    D_loss_fake_total = 0
    G_loss_total = 0
    t = tqdm.tqdm(train_loader)

    for it, (X_real, labels) in enumerate(t):
        X_real = X_real.float().to(device)
        labels = labels.to(device)

        z = torch.randn(X_real.size(0), Z_dim).to(device)
        fake_labels = torch.randint(0, num_classes, (X_real.size(0),)).to(device)

        ones_label = torch.ones(X_real.size(0), 1).to(device)
        zeros_label = torch.zeros(X_real.size(0), 1).to(device)

        # Train Discriminator
        G_sample = G(z, fake_labels)
        D_real = D(X_real, labels)
        D_fake = D(G_sample.detach(), fake_labels)

        D_loss_real = loss_fn(D_real, ones_label)
        D_loss_fake = loss_fn(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake
        D_loss_real_total += D_loss_real.item()
        D_loss_fake_total += D_loss_fake.item()

        D_solver.zero_grad()
        D_loss.backward()
        D_solver.step()

        # Train Generator
        z = torch.randn(X_real.size(0), Z_dim).to(device)
        fake_labels = torch.randint(0, num_classes, (X_real.size(0),)).to(device)
        G_sample = G(z, fake_labels)
        D_fake = D(G_sample, fake_labels)

        G_loss = loss_fn(D_fake, ones_label)
        G_loss_total += G_loss.item()

        G_solver.zero_grad()
        G_loss.backward()
        G_solver.step()

    D_loss_real_avg = D_loss_real_total / len(train_loader)
    D_loss_fake_avg = D_loss_fake_total / len(train_loader)
    D_loss_avg = D_loss_real_avg + D_loss_fake_avg
    G_loss_avg = G_loss_total / len(train_loader)

    wandb.log({
        "D_loss_real": D_loss_real_avg,
        "D_loss_fake": D_loss_fake_avg,
        "D_loss": D_loss_avg,
        "G_loss": G_loss_avg
    })

    return G, D, G_loss_avg, D_loss_avg

# Save samples function (optional)
def save_sample(G, epoch, mb_size, Z_dim):
    out_dir = "out_cgan"
    G.eval()
    with torch.no_grad():
        z = torch.randn(mb_size, Z_dim).to(device)
        labels = torch.randint(0, 10, (mb_size,)).to(device)
        samples = G(z, labels).detach().cpu().numpy()[:16]

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if not os.path.exists(f'{out_dir}'):
        os.makedirs(f'{out_dir}')

    plt.savefig(f'{out_dir}/{str(epoch).zfill(3)}.png', bbox_inches='tight')
    plt.close(fig)

# Visualize generated digits 0–9
def plot_generated_digits(G, Z_dim, device, num_images_per_class=5):
    G.eval()
    fig, axes = plt.subplots(10, num_images_per_class, figsize=(num_images_per_class, 10))
    for label in range(10):
        z = torch.randn(num_images_per_class, Z_dim).to(device)
        labels = torch.full((num_images_per_class,), label, dtype=torch.long).to(device)
        with torch.no_grad():
            samples = G(z, labels).cpu().numpy()
        for i in range(num_images_per_class):
            ax = axes[label, i]
            ax.imshow(samples[i].reshape(28, 28), cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(f"{label}", fontsize=8)
    plt.tight_layout()
    plt.suptitle("Generated Digits by Class", fontsize=16, y=1.02)
    plt.show()

# Main
wandb_log = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

G = Generator(Z_dim, h_dim, X_dim, num_classes).to(device)
D = Discriminator(X_dim, h_dim, num_classes).to(device)

G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)

loss_fn = lambda preds, targets: F.binary_cross_entropy(preds, targets)

if wandb_log:
    wandb.init(project="conditional-gan-mnist")
    wandb.config.update({
        "batch_size": mb_size,
        "Z_dim": Z_dim,
        "X_dim": X_dim,
        "h_dim": h_dim,
        "lr": lr,
    })

best_g_loss = float('inf')
save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)

epochs = 50
for epoch in range(epochs):
    G, D, G_loss_avg, D_loss_avg = cGANTraining(G, D, loss_fn, train_loader)
    print(f'epoch {epoch}; D_loss: {D_loss_avg:.4f}; G_loss: {G_loss_avg:.4f}')
    if G_loss_avg < best_g_loss:
        best_g_loss = G_loss_avg
        torch.save(G.state_dict(), os.path.join(save_dir, 'G_best.pth'))
        torch.save(D.state_dict(), os.path.join(save_dir, 'D_best.pth'))
        print(f"Saved Best Models at epoch {epoch} | G_loss: {best_g_loss:.4f}")
    save_sample(G, epoch, mb_size, Z_dim)

# Final visualization
G.load_state_dict(torch.load(os.path.join(save_dir, 'G_best.pth')))
plot_generated_digits(G, Z_dim, device, num_images_per_class=5)
