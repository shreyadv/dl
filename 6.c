import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    x = np.array(x)
    e = np.exp(x - np.max(x, axis=0 if x.ndim == 1 else 1, keepdims=True))
    return e / np.sum(e, axis=0 if x.ndim == 1 else 1, keepdims=True)

def CE(y, p):
    p = np.clip(p, 1e-12, 1)
    return -np.mean(np.sum(y * np.log(p), axis=1))

x = np.linspace(-2, 2, 5)
plt.plot(x, softmax(x), 'o-')
plt.title("Softmax Activation")
plt.xlabel("Input")
plt.ylabel("Probability")
plt.grid()
plt.show()

X = np.array([[1, 2],
              [1.5, 2.5],
              [2, 3]])

W = np.array([[0.2, -0.3, 0.5],
              [-0.5, 0.7, -0.2]])

y = np.array([[0, 1, 0],
              [1, 0, 0],
              [0, 0, 1]])

p = softmax(X @ W)

loss = CE(y, p)

grad = 0.1 * (X.T @ ((p - y) / len(X)))

print("Predictions:\n", p)
print("Loss:", loss)
print("Gradient:\n", grad)
