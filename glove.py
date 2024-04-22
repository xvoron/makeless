import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm

from base import Dataset, PAD, UNK, cut_corpus, get_corpus, get_vocab
from utils import plot_embedding


def build_cooccurence_matrix(words, n_vocab, stoi, window_size=10):
    co_occurences = np.zeros((n_vocab, n_vocab))

    for i, word in tqdm.tqdm(enumerate(words), total=len(words)):
        context_indeces = list(range(max(0, i - window_size), min(len(words), i + window_size + 1)))
        context_indeces.remove(i)

        for j in context_indeces:
            context_word = words[j]
            co_occurences[stoi[word], stoi[context_word]] += 1
    return co_occurences

def data_iter(co_occurences, min_count=10):
    nonzero_indices = np.transpose(np.nonzero(co_occurences))
    for i, j in nonzero_indices:
        X_ij = co_occurences[i, j]
        if X_ij >= min_count:
            yield i, j, X_ij


def generate_batches_from_iter(data_iter, batch_size=1024):
    batch = []
    for w_i, w_j, X_ij in data_iter:
        batch.append((w_i, w_j, X_ij))
        if len(batch) == batch_size:
            w_i, w_j, X_ij = zip(*batch)
            yield (torch.tensor(w_i, dtype=torch.long),
                   torch.tensor(w_i, dtype=torch.long),
                   torch.tensor(X_ij, dtype=torch.float))
            batch = []
    
    if batch:
        w_i, w_j, X_ij = zip(*batch)
        yield (torch.tensor(w_i, dtype=torch.long),
               torch.tensor(w_j, dtype=torch.long),
               torch.tensor(X_ij, dtype=torch.float))



class Glove(nn.Module):
    def __init__(self, n_vocab: int, n_dim: int):
        super().__init__()
        self.embedding_i = nn.Embedding(n_vocab, n_dim)
        self.embedding_j = nn.Embedding(n_vocab, n_dim)
        # bias is a vector of size n_dim
        self.bias_i = nn.Embedding(n_vocab, 1)
        self.bias_j = nn.Embedding(n_vocab, 1)


    def forward(self, w_i: int, w_j: int):
        e_i = self.embedding_i(w_i) # (batch_size, n_dim)
        e_j = self.embedding_i(w_j) # (batch_size, n_dim)

        w_i_bias = self.bias_i(w_i)
        w_j_bias = self.bias_j(w_j)

        dot_product = (e_i * e_j).sum(dim=1)
        return dot_product + w_i_bias + w_j_bias


def glove_loss(y_pred, X_ij):
    x_max = 100
    alpha = 0.75
    fX_ij = lambda x: (x / x_max).float_power(alpha).clamp(0, 1)
    return torch.mean(fX_ij(X_ij) * (y_pred - torch.log(X_ij + 1))**2)



if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    import datetime

    corpus = get_corpus()
    n_vocab = 10000
    vocab, stoi, itos = get_vocab(corpus, n_vocab)
    words = cut_corpus(corpus, vocab)


    matrix = build_cooccurence_matrix(words, n_vocab, stoi, window_size=10)
    model = Glove(n_vocab, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sw = SummaryWriter(f'runs/glove_{datetime.datetime.now().isoformat()}')

    epochs = 30
    losses = []
    for epoch in tqdm.tqdm(range(epochs), total=epochs):
        l = []
        for batch in generate_batches_from_iter(data_iter(matrix), batch_size=64):
            optimizer.zero_grad()
            w_i, w_j, X_ij = batch
            y_pred = model(w_i, w_j)
            loss = glove_loss(y_pred, X_ij)
            loss.backward()
            optimizer.step()
            l.append(loss.item())
        sw.add_scalar("Loss", np.mean(l), epoch)
        losses.append(np.mean(l))

    plt.figure()
    plt.plot(losses)
    plt.figure()
    plt.imshow(matrix)


    fig = plot_embedding(words, model.embedding_i, stoi)
    fig.savefig("glove_embedding.png")
    plt.show()
