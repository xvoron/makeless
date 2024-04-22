import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import Dataset, cut_corpus, get_corpus, get_vocab, train_loop, val_loop
from utils import plot_embedding


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

    fig = plot_embedding(words, model.token_embedding_table, stoi)
    fig.savefig("bigram_nnlm.png")
    plt.show()

