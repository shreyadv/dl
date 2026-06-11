import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

x = torch.randn(200, 2)
y = (x[:, 0] + x[:, 1] > 0).float().view(-1, 1)

class SimpleNN(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            act,
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train(lr, act, epochs=50):
    model = SimpleNN(act)
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    losses = []

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return losses

print("Learning rates...")

lrs = [0.001, 0.01, 0.1]

for lr in lrs:
    plt.plot(train(lr, nn.ReLU()), label=f"LR={lr}")

plt.legend()
plt.title("Learning Rate (ReLU)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

print("Activation functions...")

acts = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh()
}

for name, fn in acts.items():
    plt.plot(train(0.01, fn), label=name)

plt.legend()
plt.title("Activation Functions (LR=0.01)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
