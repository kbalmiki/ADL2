import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST Data
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # -> 26x26
        self.conv2 = nn.Conv2d(32, 64, 3) # -> 24x24
        self.pool = nn.MaxPool2d(2, 2)    # -> 12x12
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))     # -> [batch, 32, 26, 26]
        x = torch.relu(self.conv2(x))     # -> [batch, 64, 24, 24]
        x = self.pool(x)                  # -> [batch, 64, 12, 12]
        x = torch.flatten(x, 1)           # -> [batch, 9216]
        x = torch.relu(self.fc1(x))       # -> [batch, 128]
        x = self.fc2(x)                   # -> [batch, 10]
        return x

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
print("Training CNN...")
for epoch in range(3):
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

#Step 2 - Adverserial Image turn 4 -> 9
def generate_adversarial_example(model, image, true_label, target_label, epsilon=0.01, steps=50):
    image = image.clone().detach().to(device)
    image.requires_grad = True
    target = torch.tensor([target_label]).to(device)

    for _ in range(steps):
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, target)
        model.zero_grad()
        loss.backward()
        image.data = image.data - epsilon * image.grad.sign()
        image.grad.zero_()

    return image.detach()

# Get an image of digit 4
for img, lbl in testloader:
    if lbl.item() == 4:
        original_image = img.to(device)
        break

adv_image = generate_adversarial_example(model, original_image, 4, target_label=9)

# Visualize
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.title("Original: 4")
plt.imshow(original_image.cpu().squeeze(), cmap="gray")
plt.subplot(1,2,2)
plt.title("Adversarial: classified as 9")
plt.imshow(adv_image.cpu().squeeze(), cmap="gray")
plt.show()

# Predict both
with torch.no_grad():
    print("Prediction before attack:", torch.argmax(model(original_image)).item())
    print("Prediction after attack:", torch.argmax(model(adv_image)).item())

#Step 3 - Random noise classified as 9
# Start from random noise
random_image = torch.randn_like(original_image, requires_grad=True, device=device)
target = torch.tensor([9]).to(device)

optimizer_noise = optim.Adam([random_image], lr=0.05)

for _ in range(100):
    optimizer_noise.zero_grad()
    output = model(random_image)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer_noise.step()

    # Keep within [0,1]
    with torch.no_grad():
        random_image.clamp_(0, 1)

# Visualize
plt.title("Random noise classified as 9")
plt.imshow(random_image.cpu().detach().squeeze(), cmap="gray")
plt.show()

with torch.no_grad():
    print("Random noise prediction:", torch.argmax(model(random_image)).item())
