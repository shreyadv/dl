import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor()
])

trainset = Subset(
    torchvision.datasets.CIFAR10(
        './data',
        train=True,
        download=True,
        transform=transform
    ),
    torch.randperm(50000)[:5000]
)

testset = Subset(
    torchvision.datasets.CIFAR10(
        './data',
        train=False,
        download=True,
        transform=transform
    ),
    torch.randperm(10000)[:1000]
)

train_loader = DataLoader(trainset, batch_size=512, shuffle=True)
test_loader = DataLoader(testset, batch_size=128)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.f = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.f(x)
        return self.fc(x)

model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):

    model.train()
    total_loss = 0

    for x, y in train_loader:

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)

        loss = loss_fn(out, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for x, y in test_loader:

        x, y = x.to(device), y.to(device)

        pred = model(x).argmax(1)

        correct += (pred == y).sum().item()

        total += y.size(0)

print("Test Accuracy:", round(100 * correct / total, 2), "%")
