import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

data = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(data, range(10000)),
    batch_size=256,
    shuffle=True
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 13 * 13, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_model(opt_name):
    model = Net()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = (
        torch.optim.Adam(model.parameters(), lr=0.001)
        if opt_name == "Adam"
        else torch.optim.SGD(model.parameters(), lr=0.01)
    )

    history = []

    for epoch in range(5):
        total_loss = 0

        for x, y in loader:
            optimizer.zero_grad()

            loss = loss_fn(model(x), y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        history.append(avg_loss)

        print(f"{opt_name} Epoch {epoch+1}: {avg_loss:.4f}")

    return history

adam = train_model("Adam")
sgd = train_model("SGD")

plt.plot(adam, label="Adam")
plt.plot(sgd, label="SGD")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Adam vs SGD")
plt.legend()

plt.show()
