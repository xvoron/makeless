import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from base import Dataset, PAD, UNK, get_classification_data, get_corpus, get_vocab
from models.nnlm import NNLM, Config


def sequence(seq: list[str], stoi, max_len):
    seq = [stoi.get(word, stoi[UNK]) for word in seq]

    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [stoi[PAD]] * (max_len - len(seq))
    return seq

def batchify(X, Y, stoi, max_len):
    XX, YY = [], []
    for text, label in zip(X, Y):
        XX.append(sequence(text, stoi, max_len))
        YY.append(label)
    return torch.tensor(XX), torch.tensor(YY)



class RNNModelClassification(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def load_embeddings(self, embedding_weights: torch.Tensor, freeze: bool = True):
        self.embedding.weight = embedding_weights
        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        hidden = hidden.squeeze(0)
        return self.layers(hidden)
    

class ModelClassification(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def load_embeddings(self, embedding_weights: torch.Tensor, freeze: bool = True):
        self.embedding.weight = embedding_weights
        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = self.embedding(x) # (B, S) -> (B, S, E)
        x = x.mean(dim=1) # (B, S, E) -> (B, E)
        return self.layers(x) # (B, E) -> (B, C)

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    import datetime

    model_name = "mean"

    # load model from state dict, path nnlm.pt
    cfg = Config()
    model = NNLM(cfg.vocab_size, cfg.block_size, cfg.embedding_dim, cfg.hidden_dim)
    model.load_state_dict(torch.load("nnlm.pth"))

    vocab_size = cfg.vocab_size
    corpus = get_corpus()
    vocab, stoi, itos = get_vocab(corpus, vocab_size)

    X, Y, ltoi = get_classification_data()
    Y = [ltoi[y] for y in Y]
    num_classes = len(ltoi)


    X, Y = batchify(X, Y, stoi, 10)

    if model_name == "rnn":
        model = RNNModelClassification(vocab_size=vocab_size,
                                       embedding_dim=cfg.embedding_dim,
                                       hidden_dim=cfg.hidden_dim, num_classes=num_classes)
    else:
        model = ModelClassification(vocab_size=len(vocab),
                                    embedding_dim=cfg.embedding_dim,
                                    hidden_dim=cfg.hidden_dim, num_classes=num_classes)

    model.load_embeddings(model.embedding.weight, freeze=True)

    sw = SummaryWriter(f'runs/nnlm_classification_{model_name}_{datetime.datetime.now().isoformat()}')


    dataset = Dataset(X, Y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses = []
    accuracy = []
    for x, y in loader:
        with torch.no_grad():
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            accuracy.append(sum([1 if torch.argmax(y_pred) == y else 0 for y_pred, y in zip(y_pred, y)])/len(y))
    print(f"Loss on start: {sum(losses)/len(losses)}")
    print(f"Accuracy on start: {sum(accuracy)/len(accuracy)}")

    for e in range(50):
        losses = []
        accuracy = []
        for x, y in loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accuracy.append(sum([1 if torch.argmax(y_pred) == y else 0 for y_pred, y in zip(y_pred, y)])/len(y))
        print(f"Loss at epoch {e}: {sum(losses)/len(losses)}")
        sw.add_scalar("Loss", sum(losses)/len(losses), e)
        sw.add_scalar("Accuracy", sum(accuracy)/len(accuracy), e)

    losses = []
    accuracy = []
    for x, y in loader:
        with torch.no_grad():
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            accuracy.append(sum([1 if torch.argmax(y_pred) == y else 0 for y_pred, y in zip(y_pred, y)])/len(y))
    print(f"Loss on end: {sum(losses)/len(losses)}")
    print(f"Accuracy on end: {sum(accuracy)/len(accuracy)}")
