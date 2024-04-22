import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm

from dataset import get_data


def get_skip_grams(data: list[str], s2i: dict[str, int], window_size: int = 2):
    X, Y = [], []
    skip_grams = []
    for word in data:
        for i in range(len(word)):
            skip_gram = []
            left = max(i - window_size, 0)
            right = min(i + window_size, len(word))
            for j in range(left, right):
                if i != j:
                    skip_gram.append([word[i], word[j]])
            skip_grams.extend(skip_gram)

    X = [s2i[x[0]] for x in skip_grams]
    Y = [s2i[x[1]] for x in skip_grams]
    return torch.Tensor(X).long(), torch.Tensor(Y).long()


class Word2VecDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


with open("datasets/tinyshakespeare.txt", "r") as f:
    text = f.read()


# words = ['abc', 'bar']
char_int_map = sorted(set(''.join(words)))
s2i = {c: i+1 for i, c in enumerate(char_int_map)}
s2i['.'] = 0
i2s = {i: c for c, i in s2i.items()}
vocab_size = len(s2i) + 1

# Training split, val split, test split
# 80%, 10%, 10%

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
window_size = 4

datasets = {
        'train': get_skip_grams(words[:n1], s2i, window_size),
        'val': get_skip_grams(words[n1:n2], s2i, window_size),
        'test': get_skip_grams(words[n2:], s2i, window_size),
        }


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=20):
        super().__init__()
        self.C = nn.Embedding(vocab_size, embedding_dim)

        self.layers = nn.Sequential(*[
                nn.Linear(embedding_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, vocab_size)
            ])

    def forward(self, x):
        x = self.C(x)
        return self.layers(x)

class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        # Is this necessary?
        self.in_embed.weight
        self.out_embed.weight

    def forward(self, target, context, noise_words):

        # (batch_size) -> (batch_size, embedding_dim)
        target_embedding = self.in_embed(target)
        # (batch_size) -> (batch_size, embedding_dim)
        context_embedding = self.out_embed(context)

        # (batch_size * num_samples) -> (batch_size * num_samples, embedding_dim)
        noise_embedding = self.out_embed(noise_words)

        B, E = target_embedding.shape
        target_embedding = target_embedding.view(B, E, 1)
        context_embedding = context_embedding.view(B, 1, E)

        # (batch_size, 1, embedding_dim) * (batch_size, embedding_dim, 1) -> (batch_size, 1, 1)
        context_loss = torch.bmm(context_embedding, target_embedding)
        # (batch_size, 1, 1) -> (batch_size, 1)
        context_loss = context_loss.sigmoid().log().squeeze()

        # (batch_size * num_samples, embedding_dim) -> (batch_size, num_samples, embedding_dim)
        noise_embedding = noise_embedding.view(B, -1, E)
        # (batch_size, num_samples, embedding_dim) * (batch_size, embedding_dim, 1) -> (batch_size, num_samples, 1)
        noise_loss = torch.bmm(noise_embedding.neg(), target_embedding).sigmoid().log()
        # (batch_size, num_samples, 1) -> (batch_size, num_samples)
        noise_loss = noise_loss.squeeze().sum(1)

        return -(context_loss + noise_loss).mean()

    def get_embedding(self, input_words):
        return self.in_embed(input_words)



X, Y = datasets['train']
X_val, Y_val = datasets['val']

batch_size = 32

train = Word2VecDataset(X, Y)
val = Word2VecDataset(X_val, Y_val)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

embedding_dim = 16
hidden_dim = 24

# model = Word2VecModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
model = SkipGramNegativeSampling(vocab_size=vocab_size, embedding_dim=embedding_dim)

model.train()

loss_fn = None
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1

# losses = []
# for x, y in val_loader:
# 
#     with torch.no_grad():
#         y_hat = model(x)
#         loss = loss_fn(y_hat, y)
#         losses.append(loss.item())

# print(f"Validation loss on start: {np.mean(losses)}")

noise_dist = torch.ones(vocab_size)
num_samples = 5

for epoch in tqdm.tqdm(range(epochs)):
    for target, context in train_loader:
        optimizer.zero_grad()
        noise_words = torch.multinomial(noise_dist,
                                        batch_size*num_samples,
                                        replacement=True)


        loss = model(target=target, context=context, noise_words=noise_words)
        # loss = loss_fn(context_scores, negative_scores)
        loss.backward()
        optimizer.step()
        print(loss.item())


#     losses = []
#     for x, y in val_loader:
#         with torch.no_grad():
#             y_hat = model(x)
#             loss = loss_fn(y_hat, y)
#             losses.append(loss.item())
# 
#     print(f"Validation loss on {epoch}: {np.mean(losses)}")

# losses = []
# for x, y in val_loader:
#     with torch.no_grad():
#         y_hat = model(x)
#         loss = loss_fn(y_hat, y)
#         losses.append(loss.item())

# print(f"Validation loss at the end: {np.mean(losses)}")
