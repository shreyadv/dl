import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text = "pytorch is great for building deep learning models and rnn models are fun"

words = text.split()

vocab = sorted(set(words))

w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}

X = torch.LongTensor([w2i[w] for w in words[:-1]]).unsqueeze(1).to(device)
Y = torch.LongTensor([w2i[w] for w in words[1:]]).to(device)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        out, _ = self.rnn(self.emb(x))
        return self.fc(out.squeeze(1))

model = RNNModel(len(vocab)).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.05)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):

    optimizer.zero_grad()

    loss = loss_fn(model(X), Y)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

def predict(word):

    model.eval()

    with torch.no_grad():

        idx = torch.LongTensor([[w2i[word]]]).to(device)

        pred = model(idx).argmax(1).item()

        return i2w[pred]

print("After 'pytorch' -->", predict("pytorch"))
print("After 'building' -->", predict("building"))
