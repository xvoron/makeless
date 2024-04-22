"""
This is implementation of Bengio's NNLM (Neural Network Language Model) using PyTorch.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from base import Dataset, cut_corpus, get_corpus, get_vocab, train_loop, val_loop
from utils import plot_embedding


@dataclass
class Config:
    vocab_size: int = 10000
    embedding_dim: int =100
    hidden_dim: int = 100
    epochs: int = 5
    batch_size: int = 128
    block_size: int = 3


def build_dataset(words, s2i, block_size=3):
    X, Y = [], []
    for idx in range(block_size, len(words)):
        context = words[idx-block_size:idx]
        X.append([s2i[word] for word in context])
        Y.append(s2i[words[idx]])
    return torch.tensor(X), torch.tensor(Y)


class NNLM(nn.Module):
    def __init__(self, vocab_size: int, block_size: int,
                 embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # (V, D)

        self.layers = nn.Sequential(*[
                nn.Linear(block_size * embedding_dim, hidden_dim, bias=False),
                nn.Tanh(),
                nn.Linear(hidden_dim, vocab_size),
                nn.LogSoftmax(dim=1)
                ])

    def forward(self, x: torch.Tensor):
        # (batch_size, block_size)
        x = self.embedding(x) # (batch_size, block_size, embedding_dim)
        x = x.reshape(x.shape[0], -1) # (batch_size, block_size * embedding_dim)
        x = self.layers(x) # (batch_size, vocab_size)
        return x


if __name__ == "__main__":

    cfg = Config()

    corpus = get_corpus()
    vocab, stoi, itos = get_vocab(corpus, cfg.vocab_size)
    words = cut_corpus(corpus, vocab)

    # random.seed(42)
    # random.shuffle(words)
    # n1 = int(0.8*len(words))
    # n2 = int(0.9*len(words))

    # train = build_dataset(words[:n1], stoi, block_size=cfg.block_size)
    # val = build_dataset(words[n1:n2], stoi, block_size=cfg.block_size)
    # test = build_dataset(words[n2:], stoi, block_size=cfg.block_size) 
    train = build_dataset(words, stoi, block_size=cfg.block_size)

    model = NNLM(cfg.vocab_size, cfg.block_size, cfg.embedding_dim, cfg.hidden_dim)
    dataloader = torch.utils.data.DataLoader(Dataset(*train), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    # val_dataloader = torch.utils.data.DataLoader(Dataset(*val), batch_size=32, shuffle=True, drop_last=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # print(f"Validation loss before training: {val_loop(model, val_dataloader, loss_fn)}")
    # train_loop(model, dataloader, loss_fn, optimizer, cfg.epochs, val_dataloader)
    # print(f"Validation loss after training: {val_loop(model, val_dataloader, loss_fn)}")
    train_loop(model, dataloader, loss_fn, optimizer, cfg.epochs, log_every_n_step=300, experiment_name="nnlm")

    fig = plot_embedding(words, model.embedding, stoi)
    fig.savefig("nnlm_embedding.png")
    plt.show()

    # save model
    torch.save(model.state_dict(), "nnlm.pth")
