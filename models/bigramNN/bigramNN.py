"""
This is single file implementation of a bigram neural network language model.

# Bi-gram Neural Network Language Model.


## Intro 
Bi-gram language model is a simple language model based on the probability of a word given the
previous word. The neural network version of the model is trying to learn the embeddings of the
words in the corpus using gradient descent. 
"""
from collections import Counter
import datetime
import re

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm


UNK = '<unk>'
PAD = '<pad>'

STOI = dict[str, int]
ITOS = dict[int, str]


def text_preprocessor(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    return re.sub(r'\s+', ' ', text)


def get_corpus() -> str:
    """Get the corpus of the data as a string.

    Note:
        Use only the `rss_title` column.
    """
    data = pl.read_csv('./dataset/sportoclanky.csv')
    data = data.select([
        data['rss_title'],
    ])
    corpus = []
    for row in data.iter_rows():
        corpus.append(row[0])

    return text_preprocessor(' '.join(corpus))


def get_vocab(corpus: str, vocab_size: int = 10000) -> tuple[list[str], STOI, ITOS]:
    counts =  Counter(corpus.split())
    vocab = [word for word, _ in counts.most_common(vocab_size - 2)]
    vocab = list(set(vocab)) + [PAD, UNK]
    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for word, i in stoi.items()}
    return vocab, stoi, itos


def cut_corpus(corpus: str, vocab: list[str]) -> list[str]:
    return [word if word in vocab else UNK for word in corpus.split()]



def train_loop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    val_loader: torch.utils.data.DataLoader | None = None, 
    log_every_n_step: int = 100,
    experiment_name: str = 'default'
):
    sw = SummaryWriter(f'runs/{experiment_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    step = 0

    loss_tracker = []
    for e in tqdm.tqdm(range(epochs)):

        for x, y in train_loader:
            optimizer.zero_grad()
            prediction = model(x)
            loss = loss_fn(prediction, y)
            loss_tracker.append(loss.item())
            loss.backward()
            optimizer.step()

            step += 1

            if step % log_every_n_step == 0:
                print(f"Loss at step {step}: {np.mean(loss_tracker)}")
                sw.add_scalar('Loss/train', np.mean(loss_tracker), step)
                loss_tracker = []

                if val_loader:
                    val_loss = val_loop(model, val_loader, loss_fn)
                    sw.add_scalar('Loss/val', val_loss, step)


def val_loop(model: nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: nn.Module
             ):
    loss_tracker = []
    for x, y in dataloader:
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss_tracker.append(loss.item())
    return np.mean(loss_tracker)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


class BigramNNLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor):
        return self.token_embedding_table(idx)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx) # (B, T, C)
            logits = logits[:, -1, :]  # Taking only the last token (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
        return idx


if __name__ == "__main__":
    corpus = get_corpus()
    vocab, stoi, itos = get_vocab(corpus, 10000)
    words = cut_corpus(corpus, vocab)

    X, Y = zip(*list(zip(words, words[1:])))
    X = torch.tensor([stoi[x] for x in X])
    Y = torch.tensor([stoi[y] for y in Y])

    dataset = Dataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    model = BigramNNLM(vocab_size=len(vocab))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    loss_fn = nn.CrossEntropyLoss()

    train_loop(
            model=model,
            train_loader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=2,
            experiment_name="bigram_nnlm",
            )

    start_word = 'krčmář'
    data = torch.tensor([stoi[start_word]]).unsqueeze(0)
    print([itos[int(d)] for d in model.generate(data, 10).squeeze(0)])

    # fig = plot_embedding(words, model.token_embedding_table, stoi)
    # fig.savefig("bigram_nnlm.png")
    # plt.show()
