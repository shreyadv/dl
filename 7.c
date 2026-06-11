import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

t = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=t
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=t
)

train = torch.utils.data.DataLoader(
    torch.utils.data.Subset(trainset, range(10000)),
    batch_size=64,
    shuffle=True
)

test = torch.utils.data.DataLoader(
    torch.utils.data.Subset(testset, range(5000)),
    batch_size=64
)

class Net(nn.Module):
    def __init__(self, bn=False, do=False):
        super().__init__()

        self.c = nn.Conv2d(3, 16, 3, 1, 1)

        self.bn = nn.BatchNorm2d(16) if bn else nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)

        self.drop = nn.Dropout(0.3) if do else nn.Identity()

        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn(self.c(x))))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def run(name, model, epochs=5):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()

        for x, y in train:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    print(name, "Accuracy:", round(correct / total, 4))

run("Base", Net())
run("BatchNorm", Net(bn=True))
run("Dropout", Net(do=True))
run("BN + Dropout", Net(bn=True, do=True))
