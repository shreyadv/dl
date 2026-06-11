import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

class NN:
    def __init__(self, lr=0.1, loss='mse'):
        np.random.seed(42)
        self.W1 = np.random.randn(2,4) * 0.5
        self.b1 = np.zeros((1,4))
        self.W2 = np.random.randn(4,1) * 0.5
        self.b2 = np.zeros((1,1))
        self.lr = lr
        self.loss = loss

    def sig(self, x, d=False):
        if d:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def lossf(self, o, y):
        e = 1e-8
        if self.loss == 'mse':
            return np.mean((o - y) ** 2)
        return -np.mean(y*np.log(o+e) + (1-y)*np.log(1-o+e))

    def dloss(self, o, y):
        e = 1e-8
        if self.loss == 'mse':
            return (o - y)
        return (o - y) / ((o + e) * (1 - o + e))

    def fwd(self, X):
        self.a1 = self.sig(X @ self.W1 + self.b1)
        self.a2 = self.sig(self.a1 @ self.W2 + self.b2)
        return self.a2

    def train(self, X, y, ep=500):
        for i in range(ep):
            o = self.fwd(X)

            dz2 = self.dloss(o, y) * self.sig(self.a2, 1)
            dz1 = (dz2 @ self.W2.T) * self.sig(self.a1, 1)

            self.W2 -= self.lr * (self.a1.T @ dz2) / len(X)
            self.b2 -= self.lr * np.mean(dz2, axis=0, keepdims=True)

            self.W1 -= self.lr * (X.T @ dz1) / len(X)
            self.b1 -= self.lr * np.mean(dz1, axis=0, keepdims=True)

    def acc(self, X, y):
        return np.mean((self.fwd(X) > 0.5) == y) * 100

for lr in [0.01, 0.1]:
    print(f"\nLearning Rate = {lr}")
    nn = NN(lr, 'mse')
    nn.train(X, y)
    print("Accuracy:", nn.acc(X, y))

for l in ['mse', 'bce']:
    print(f"\nLoss Function = {l.upper()}")
    nn = NN(0.1, l)
    nn.train(X, y)
    print("Accuracy:", nn.acc(X, y))
