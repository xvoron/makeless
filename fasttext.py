from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from base import Dataset, PAD, UNK, cut_corpus, get_corpus, get_vocab


SOF = "<"
EOF = ">"

def get_subwords(sample):
    return list(map(lambda w: wordix_to_subix[w], sample))


class FastText(nn.Module):
    def __init__(self, n_subvocab: int, n_vocab: int, n_grams: int, n_dim: int):
        super().__init__()
        self.embedding_z = nn.EmbeddingBag(n_subvocab + 1, embedding_dim=n_dim, mode="sum")
        self.embedding_v = nn.Embedding(n_vocab, n_dim)

    def forward(self, x):
        print(x.shape)
        x_1, x_2 = x.T
        k = x_1.shape[0]
        x_1_sub = torch.LongTensor([get_subwords(x_1[i]) for i in range(k)])
        u = torch.stack([self.embedding_z(sub) for sub in x_1_sub])
        v = self.embedding_v(x_2)
        y = (u * v).sum(dim=2).T
        return y

def subsample(words: list[str], threshold: float =1e-5) -> list[str]:
    """
    In original paper implementation is different:
        P(w_i) = (sqrt(z(w_i)/0.001) + 1) * (0.001/z(w_i))

    """
    words, counts = zip(*Counter(words).items())

    freqs = {word: count/len(words) for word, count in zip(words, counts)}

    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in words}

    # 1 - p_drop is the probability of keeping the word
    # so if np.random.random() = 0.8 and p_drop = 0.5 (1 - 0.5 = 0.5) we keep the word
    return [word for word in words if np.random.random() < (1 - p_drop[word])]

corpus = get_corpus()[:10000]

n_vocab = 1000

vocab, stoi, itos = get_vocab(corpus, n_vocab)
words = cut_corpus(corpus, vocab)

words = subsample(words)

def n_grams(word: str, n: int = 3) -> list[str]:
    return [word[i:i+n] for i in range(len(word)-n+1)] if word != UNK else []


def get_context(words, idx, window_size=5):
    """
    words = [0, 1, 2, 3, 4, 5, 6]
    idx = 2
    window_size = 2

    R = 1 -> return [1, 3]
    R = 2 -> return [0, 1, 3, 4]

    So the output is always symmetric around the index, except the index itself
    is not included and beginning and the end of the list are edge cases.

    """
    R = np.random.randint(1, window_size+1)
    start = idx - R if idx - R > 0 else 0
    stop = idx + R
    return words[start:idx] + words[idx+1:stop+1]

def get_xy(words, window_size=5):
    X, Y = [], []
    for i, word in enumerate(words):
        for context in get_context(words, i, window_size):
            X.append(word)
            Y.append(context)
    return X, Y

X, Y = get_xy(words, window_size=5)
X = torch.LongTensor([stoi[x] for x in X])
Y = torch.LongTensor([stoi[y] for y in Y])

vocab = [SOF + word + EOF for word in vocab]
words = [SOF + word + EOF for word in words]

def split_to_subword(words, subword_len=3):
    subwords = []
    for w in words:
        subwords.extend(n_grams(w, subword_len))
    return subwords

subvocab = list(set(split_to_subword(vocab)))
print(subvocab[:5])
stoi_sub = {k: i for i, k in enumerate(subvocab)}
itos_sub = {i: k for i, k in enumerate(subvocab)}

subword_idx = [list(map(lambda x: stoi_sub[x], n_grams(word))) for word in vocab]
wordix_to_subix = {wordix: subix for wordix, subix in enumerate(subword_idx)}

print(list(wordix_to_subix.items())[:5])

padding = max([len(s) for s in wordix_to_subix.values()])
# This can be problem that there are different vocabs here
wordix_to_subix = {wordix: subix + [stoi[PAD]] * (padding - len(subix)) for wordix, subix in enumerate(subword_idx)}
print(list(wordix_to_subix.items())[:5])


model = FastText(len(subvocab), len(vocab), 3, 100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

dataset = Dataset(X, Y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

losses = []
epochs = 10
for e in range(epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {e+1}/{epochs}, Loss: {losses[-1]}")
