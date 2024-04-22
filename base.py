from collections import Counter
import datetime
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

UNK = '<unk>'
PAD = '<pad>'

def get_data():
    return pl.read_csv('./dataset/sportoclanky.csv')

def get_corpus():
    """group rss_title and rss_perex into one column"""
    data = get_data()
    data = data.select([
        data['rss_title'],
    ])
    corpus = []
    for row in data.iter_rows():
        corpus.append(row[0])

    return preprocess(' '.join(corpus))


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    return re.sub(r'\s+', ' ', text)


def get_vocab(corpus: str, vocab_size: int = 10000):
    """get the vocabulary of the corpus"""
    counts =  Counter(corpus.split())
    vocab = [word for word, _ in counts.most_common(vocab_size - 2)]
    vocab = list(set(vocab))
    vocab.append(PAD)
    vocab.append(UNK)
    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for word, i in stoi.items()}
    return vocab, stoi, itos


def get_classification_data(first_n: Optional[int] = None):
    data = get_data()
    data = data.select([
        data['category'],
        data['rss_title'],
    ])
    X, Y = [], []

    data_to_iter = data.iter_rows() if first_n is None else data.iter_rows()[:first_n]
    for row in data_to_iter:
        X.append(preprocess(row[1]).split())
        Y.append(preprocess(row[0]))

    ltoi = {label: i for i, label in enumerate(set(Y))}

    return X, Y, ltoi

def cut_corpus(corpus: str, vocab: list[str]):
    return [word if word in vocab else UNK for word in corpus.split()]


def visualize_embeddings(C, i2s):
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), i2s[i], ha='center', va='center', color='black')
    plt.grid(True, 'minor')
    plt.show()



def train_loop(model: nn.Module,
               train_loader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               epochs: int,
               val_loader: Optional[torch.utils.data.DataLoader] = None, 
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


if __name__ == '__main__':
    data = get_data()
    print(data)
    corpus = get_corpus()
    print(len(corpus.split()))
