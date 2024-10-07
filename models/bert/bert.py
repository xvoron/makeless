from __future__ import annotations
import re

import numpy as np
import torch


UNK = '<unk>'
PAD = '<pad>'
MASK = '<mask>'
SEP = '<sep>'
CLS = '<cls>'
KEEP = '<keep>'
RAND = '<rand>'
ORIG = '<orig>'
SPEC = [UNK, PAD, MASK, SEP, CLS, KEEP, RAND, ORIG]



class TinyTokenizer:
    """This tokenizer is a simple character level tokenizer, with extension for special tokens.

    So vocab is equal to all characters utf-8 (256 characters) plus special tokens.
    Special tokens are handled by the tokenizer itself.
    """
    def __init__(self, block_size: int, special_tokens: list[str] = []):
        self.vocab = {idx: bytes([idx]).decode('utf-8') for idx in range(128)}
        self.block_size = block_size
        self.special_tokens = special_tokens

        self.vocab.update({
            len(self.vocab) + idx: special_token
            for idx, special_token in enumerate(special_tokens)
        })
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text: str):
        # split by self.special_tokens
        special_tokens = re.split(f"({'|'.join(self.special_tokens)})", text)

        tokens = []
        for token in special_tokens:
            if token in self.special_tokens:
                tokens.append(self.inv_vocab[token])
            else:
                tokens.extend([self.inv_vocab.get(c, self.inv_vocab[UNK]) for c in token])

        return tokens

    def detokenize(self, tokens: list[int]):
        return ''.join([self.vocab[token] for token in tokens])


class MLM:
    def __init__(
        self,
        corpus: list[int],
        vocab: dict[int, str],
        mask_token: str = MASK,
        mask_prob: float = 0.15,
    ):
        self.corpus = corpus
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.mask_token = mask_token
        self.mask_prob = mask_prob

    def mask(self, tokens: list[int]):
        tokens_to_mask = np.random.rand(len(tokens)) < self.mask_prob
        ids_to_mask = np.where(tokens_to_mask)[0]

        tokens_new = tokens.copy()
        # Default label is -100, which is ignored by the CrossEntropyLoss
        labels = [-100] * len(tokens)

        for token_id in ids_to_mask:
            labels[token_id] = tokens[token_id]  # Set the label

            rand = np.random.rand()
            if rand < 0.8:
                tokens_new[token_id] = self.inv_vocab[MASK]  # Replace with [MASK]
            elif rand < 0.9:
                tokens_new[token_id] = int(np.random.choice(list(self.vocab.keys())))  # Random token
            else:
                tokens_new[token_id] = tokens[token_id]  # Keep the original token

        return tokens_new, labels



class Head(torch.nn.Module):
    """This is BERT Head module

    The head module is used to calculate the attention weights for the
    multi-head attention mechanism.
    """
    def __init__(self, n_embed: int,  head_size: int, mask: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.key = torch.nn.Linear(n_embed, head_size)
        self.query = torch.nn.Linear(n_embed, head_size)
        self.value = torch.nn.Linear(n_embed, head_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor):
        _, T, C = x.shape

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        dk = C ** 0.5

        weight = query @ key.transpose(-2, -1) / dk

        # Adjust the mask to the current sequence length
        mask = self.mask[:T, :T]
        weight = weight.masked_fill(mask == 0, float('-inf'))
        weight = torch.nn.functional.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        return weight @ value


class MultiHeadAttention(torch.nn.Module):
    """This is BERT MultiHeadAttention module

    The MultiHeadAttention module is used to calculate the attention weights
    for the multi-head attention mechanism.
    """
    def __init__(self, num_heads: int, head_size: int, n_embed: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        # Create a mask for the attention mechanism
        # in BERT model, the model can see the future so the mask is an 
        # ones matrix
        mask = torch.ones(block_size, block_size)
        self.heads = torch.nn.ModuleList([Head(n_embed, head_size, mask, dropout) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(num_heads * head_size, n_embed)

    def forward(self, x: torch.Tensor):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(output)


class FeedForward(torch.nn.Module):
    def __init__(self, n_embed: int, dropout: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embed, n_embed),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(torch.nn.Module):
    def __init__(self, n_embed: int, n_heads: int, block_size: int):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size, n_embed, block_size=block_size)
        self.ff = FeedForward(n_embed)
        self.ln1 = torch.nn.LayerNorm(n_embed)
        self.ln2 = torch.nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor):
        # residual connection
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyBERT(torch.nn.Module):
    def __init__(self, n_embed: int, n_heads: int, n_layers: int, vocab_size: int, block_size: int):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embed)
        self.blocks = torch.nn.ModuleList([Block(n_embed, n_heads, block_size) for _ in range(n_layers)])

        self.ln_f = torch.nn.LayerNorm(n_embed)
        self.lm_head = torch.nn.Linear(n_embed, vocab_size)

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        x = self.token_embedding_table(x)
        x = x + self.position_embedding_table(torch.arange(T, device=x.device))

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x


class TinyBERTLoss(torch.nn.Module):
    """Loss applied only to the masked tokens"""
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        return self.loss_fn(logits, labels)


def add_padding(tokens: list[int], block_size: int, pad_token: int, is_label=False):
    if len(tokens) > block_size:
        tokens = tokens[:block_size]
    elif len(tokens) < block_size:
        pad_value = pad_token if not is_label else -100
        tokens += [pad_value] * (block_size - len(tokens))
    return tokens


class Dataset(torch.utils.data.Dataset):
    def __init__(self, corpus_tokenized, block_size):
        self.corpus_tokenized = corpus_tokenized
        self.block_size = block_size

    def __len__(self):
        return len(self.corpus_tokenized) // self.block_size

    def __getitem__(self, idx):
        tokens = self.corpus_tokenized[idx * self.block_size:(idx + 1) * self.block_size]
        masked_tokens, labels = mlm.mask(tokens)

        masked_tokens = add_padding(masked_tokens, self.block_size, tokenizer.inv_vocab[PAD])
        labels = add_padding(labels, self.block_size, tokenizer.inv_vocab[PAD], is_label=True)

        x = torch.tensor(masked_tokens, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y


if __name__ == "__main__":

    from pathlib import Path
    # corpus_path = Path('../../dataset/tinyshakespeare.txt')
    corpus_path = Path('dataset/tinyshakespeare.txt')
    corpus = corpus_path.read_text()

    # BERT model
    n_embed = 128
    n_heads = 8
    n_layers = 4
    block_size = 256
    epochs = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = TinyTokenizer(special_tokens=SPEC, block_size=block_size)
    vocab_size = len(tokenizer.vocab)
    corpus_tokenized = tokenizer.tokenize(corpus)
    mlm = MLM(corpus_tokenized, vocab=tokenizer.vocab, mask_prob=0.15)

    tiny_bert = TinyBERT(n_embed, n_heads, n_layers, vocab_size, block_size).to(device)
    loss_fn = TinyBERTLoss().to(device)
    optimizer = torch.optim.Adam(tiny_bert.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 80 % of the corpus for training
    train_corpus_tokenized = corpus_tokenized[:int(0.8 * len(corpus_tokenized))]
    valid_corpus_tokenized = corpus_tokenized[int(0.8 * len(corpus_tokenized)):]

    train = Dataset(corpus_tokenized, block_size)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True
    )
    valid = Dataset(valid_corpus_tokenized, block_size)
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True
    )

    for epoch in range(epochs):
        train_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = tiny_bert(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        for i, (x, y) in enumerate(valid_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                logits = tiny_bert(x)
                loss = loss_fn(logits, y)
                valid_loss += loss.item()

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} - Valid loss: {valid_loss:.4f}")

        scheduler.step()

    test_text = "a<mask><mask> yo<mask>"
    test_tokens_orig = tokenizer.tokenize(test_text)
    test_tokens = [tokenizer.inv_vocab[CLS]] + test_tokens_orig + [tokenizer.inv_vocab[SEP]]
    test_tokens = add_padding(test_tokens, block_size, tokenizer.inv_vocab[PAD])
    x = torch.tensor(test_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Identify mask token positions
    mask_token_id = tokenizer.inv_vocab[MASK]
    mask_positions = [i for i, token_id in enumerate(test_tokens) if token_id == mask_token_id]

    # Get predictions
    with torch.no_grad():
        logits = tiny_bert(x)
    predicted_ids = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    # Replace only the [MASK] tokens in the original tokens
    for pos in mask_positions:
        test_tokens[pos] = predicted_ids[pos]

    # Detokenize
    predicted_text = tokenizer.detokenize(test_tokens[1:len(test_tokens_orig)+1])  # Exclude CLS and PAD tokens
    print("Predicted:", predicted_text)
    print("Original:", test_text)
