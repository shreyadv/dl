import torch
import torch.nn as nn
import torch.optim as optim

text = "to be or not to be that is the question".split()

vocab = sorted(set(text))

w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}

seq_len = 3

X, Y = [], []

for i in range(len(text) - seq_len):
    X.append([w2i[w] for w in text[i:i + seq_len]])
    Y.append(w2i[text[i + seq_len]])

X = torch.tensor(X)
Y = torch.tensor(Y)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb = nn.Embedding(len(vocab), 32)

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, len(vocab))

    def forward(self, x):
        x, _ = self.lstm(self.emb(x))
        return self.fc(x[:, -1])

model = Model()

optimizer = optim.Adam(model.parameters(), lr=0.01)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):

    optimizer.zero_grad()

    loss = loss_fn(model(X), Y)

    loss.backward()

    optimizer.step()

def generate(seed, n=5):

    words = seed.split()

    model.eval()

    with torch.no_grad():

        for _ in range(n):

            x = torch.tensor(
                [[w2i[w] for w in words[-seq_len:]]]
            )

            pred = model(x).argmax(1).item()

            words.append(i2w[pred])

    return " ".join(words)

print(generate("to be or"))
