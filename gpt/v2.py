from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F



block_size = 256
batch_size = 64
epoch = 5000
lr = 3e-4
max_iters = 3000
eval_interval = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_layers = 6
n_heads = 6
dropout = .2


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


class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        _, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weight = q @ k.transpose(-2, -1) * C ** -0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        v = self.value(x)

        return weight @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)

    def forward(self, x: torch.Tensor):
        output =  torch.cat([head(x) for head in self.heads], dim=-1)

        # Linear projection
        return self.proj(output)


class FeedForward(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
                # 4 * n_embed is a hyperparameter from the paper
                nn.Linear(n_embed, 4 * n_embed),
                nn.ReLU(),
                nn.Linear(4 * n_embed, n_embed),
                nn.Dropout(dropout),
                )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed: int, n_heads: int):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor):
        # residual connection
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)]) 
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Args:
            idx: torch.Tensor of shape (B, T)
            targets: torch.Tensor of shape (B, T)

        Return:
            tensor of shape (B, T, C), loss

        Where B is batch dimension, T - time dimension, C - channel dimension.
        """
        B, T = idx.shape
        # (B, T, C)
        token_emb = self.token_embedding_table(idx)

        # (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        x = token_emb + pos_emb # (B, T, C)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # (B, T, C)
            logits, _ = self(idx_cond)
            # We take only the last T 
            logits = logits[:, -1, :]  # (B, C)

            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)

        return idx

model = BigramLanguageModel().to(device)

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
