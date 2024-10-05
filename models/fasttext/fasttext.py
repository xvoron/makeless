from collections import Counter
import re

import numpy as np
import polars as pl
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

UNK = '<unk>'
PAD = '<pad>'
SOF = "<"
EOF = ">"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def get_corpus():
    """Get the corpus of the data as a string.

    Note:
        Use only the `rss_title` column.
    """
    data = pl.read_csv('dataset/sportoclanky.csv')
    data = data.select([
        data['rss_title'],
    ])
    corpus = []
    for row in data.iter_rows():
        corpus.append(row[0])

    return preprocess(' '.join(corpus))

# def get_corpus():
#     with open('./dataset/tinyshakespeare.txt', 'r') as f:
#         text = f.read()
#         return preprocess(text)

def cut_corpus(corpus: str, vocab: list[str]):
    return [word if word in vocab else UNK for word in corpus.split()]

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    return re.sub(r'\s+', ' ', text)


def get_vocab(corpus: str, vocab_size: int = 10000):
    """get the vocabulary of the corpus"""
    counts =  Counter(corpus.split())
    vocab = [word for word, _ in counts.most_common(vocab_size - 2)]
    vocab.extend([PAD, UNK])
    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for word, i in stoi.items()}
    return vocab, stoi, itos

def get_subwords(sample):
    return list(map(lambda w: wordix_to_subix[w], sample))


class FastText(nn.Module):
    def __init__(self, n_subvocab: int, n_vocab: int, n_grams: int, n_dim: int):
        super().__init__()
        self.embedding_z = nn.Embedding(n_subvocab + 1, embedding_dim=n_dim, padding_idx=stoi_sub[PAD])
        self.embedding_v = nn.Embedding(n_vocab, n_dim)

    def forward(self, x):
        # TODO: read the paper one more time

        # x: (batch_size, 2) where x[:, 0] is the word and x[:, 1] is the context.
        # split the input x into x_1 and x_2 where x_1 is the word and x_2 is the context.
        x_1, x_2 = x[:, 0], x[:, 1]
        x_1_sub = torch.LongTensor([wordix_to_subix[w.item()] for w in x_1]).to(device)
        # shape: (batch_size, padding_length)
        u_sub_emb = self.embedding_z(x_1_sub)
        u = u_sub_emb.sum(dim=1)
        v = self.embedding_v(x_2)
        y = (u * v).sum(dim=1)
        return y

def subsample(words: list[str], threshold: float =1e-5) -> list[str]:
    """
    In original paper implementation is different:
        P(w_i) = (sqrt(z(w_i)/0.001) + 1) * (0.001/z(w_i))

    """
    counts = Counter(words)
    freqs = {word: count/len(words) for word, count in counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in counts.keys()}
    # 1 - p_drop is the probability of keeping the word
    # so if np.random.random() = 0.8 and p_drop = 0.5 (1 - 0.5 = 0.5) we keep the word
    return [word for word in words if np.random.random() < (1 - p_drop[word])]


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

def get_xy_neg_sampling(words: list[str], window_size: int = 5, num_negative_samples: int = 5):
    X, Y = [], []
    vocab_indicies = list(range(len(vocab)))
    for i, word in enumerate(words):
        context_words = get_context(words, i, window_size)
        for context in context_words:
            X.append((stoi[word], stoi[context]))
            Y.append(1)
            for _ in range(num_negative_samples):
                negative_context = np.random.choice(vocab_indicies)
                X.append((stoi[word], negative_context))
                Y.append(0)
    return X, Y

def split_to_subword(words, subword_len=3):
    subwords = []
    for w in words:
        subwords.extend(n_grams(w, subword_len))
    return subwords


if __name__ == '__main__':

    corpus = get_corpus()

    n_vocab = 1000

    vocab, stoi, itos = get_vocab(corpus, n_vocab)
    words = cut_corpus(corpus, vocab)
    words = subsample(words)

    X, Y = get_xy_neg_sampling(words, window_size=5)
    X = torch.LongTensor(X)
    Y = torch.FloatTensor(Y)

    vocab_with_markers = [SOF + word + EOF for word in vocab]
    subwords = split_to_subword(vocab_with_markers)

    subvocab = list(set(subwords))
    if PAD not in subvocab:
        subvocab.append(PAD)

    stoi_sub = {k: i for i, k in enumerate(subvocab)}
    itos_sub = {i: k for i, k in enumerate(subvocab)}

    subword_idx = [list(map(lambda x: stoi_sub[x], n_grams(SOF + word + EOF))) for word in vocab]
    padding_length = max([len(s) for s in subword_idx])
    wordix_to_subix = {
        i: s + [stoi_sub[PAD]] * (padding_length - len(s))
        for i, s in enumerate(subword_idx)}

    model = FastText(n_subvocab=len(subvocab), n_vocab=len(vocab), n_grams=3, n_dim=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    dataset = Dataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    losses = []
    epochs = 10
    for e in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        sheduler.step()
        print(f"Epoch {e+1}/{epochs}, Loss: {losses[-1]}")


    def plot_embedding(words, embedding, word_to_ix, top=150):
        """
        words: raw tokens
        embedding: torch.nn.Embedding() object
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        counter = Counter(words)
        
        test_words = counter.most_common(top)
        test_words_raw = [w for w, _ in test_words]
        test_words = [word_to_ix[w] for w in test_words_raw]
        
        with torch.no_grad():
            embed_xy = embedding(torch.tensor(test_words)).detach().numpy()
            embed_xy = TSNE(n_components=2).fit_transform(embed_xy)
            embed_x, embed_y = list(zip(*embed_xy))
        
        fig = plt.figure(figsize=(10, 10))
        for xy, word in zip(embed_xy, test_words_raw):
            plt.annotate(word, xy, clip_on=True, fontsize=14)

        plt.title("Word Embedding")
        plt.scatter(embed_x, embed_y, alpha=.3)
        plt.axhline([0], ls=":", c="grey")
        plt.axvline([0], ls=":", c="grey")
        return fig
    
    model.eval()
    model.to('cpu')
    fig = plot_embedding(vocab, model.embedding_v, stoi)
    fig.savefig("embedding.png")
