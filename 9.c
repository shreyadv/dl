import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])

train = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    ),
    batch_size=128,
    shuffle=True
)

test = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transform
    ),
    batch_size=128
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.m = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),

            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.m(x)

model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):

    correct = total = 0
    loss_sum = 0

    for x, y in train:

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)

        loss = loss_fn(out, y)

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    print(
        f"Epoch {epoch+1} "
        f"Loss:{loss_sum/len(train):.4f} "
        f"Acc:{100*correct/total:.2f}%"
    )

model.eval()

correct = total = 0

with torch.no_grad():

    for x, y in test:

        x, y = x.to(device), y.to(device)

        pred = model(x).argmax(1)

        correct += (pred == y).sum().item()

        total += y.size(0)

print("Test Accuracy:", round(100 * correct / total, 2), "%")
