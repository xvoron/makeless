from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm



block_size = 8
batch_size = 64
epoch = 10000
lr = 1e-2
max_iters = 3000
eval_interval = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


torch.manual_seed(1337)


with open('./tinyshakespeare.txt', 'r') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)

n = int(.9 * len(data))
train_data = data[:n]
val_data = data[n:]

x = train_data[:block_size]
y = train_data[1:block_size + 1]


def get_batch(split_):
    data = train_data if split_ == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split_ in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split_)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_] = losses.mean().item()
    model.train()
    return out



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Args:
            idx: torch.Tensor of shape (B, T)
            targets: torch.Tensor of shape (B, T)

        Return:
            tensor of shape (B, T, C), loss

        Where B is batch dimension, T - time dimension, C - channel dimension.
        """
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)

            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx has shape (B, T)
        for _ in range(max_new_tokens):
            # (B, T, C)
            logits, _ = self(idx)
            # We take only the last T 
            logits = logits[:, -1, :]  # (B, C)

            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)

        return idx

model = BigramLanguageModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


# write loss to progress bar
# progress = tqdm(range(epoch), total=epoch, desc='Loss: ')


for step in range(epoch):
    xb, yb = get_batch('train')

    if step % eval_interval == 0:
        losses = estimate_loss()
        report = f"{step}: train {losses['train']:.4f}, val {losses['val']:.4f}"
        print(report)
        # progress.set_description(report)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=200)[0].tolist()))
