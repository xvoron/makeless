"""Implementation of the RetroMAE algorithm."""
from pathlib import Path
import re

import numpy as np
import torch


CLS = '<cls>'
SEP = '<sep>'
MASK = '<mask>'
PAD = '<pad>'

SPECIAL_TOKENS = [CLS, SEP, MASK, PAD]

def tokenize(text, inv_vocab):

    special_tokens = re.split(f"({'|'.join(SPECIAL_TOKENS)})", text)
    encoded_text = []

    for token in special_tokens:
        if token in SPECIAL_TOKENS:
            encoded_text.append(inv_vocab[token])
        else:
            chars = list(token)
            encoded_text.extend([inv_vocab[char] for char in chars])
    return encoded_text

def detokenize(encoded_text, vocab):
    return ''.join([vocab[token] for token in encoded_text])


class Head(torch.nn.Module):
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


class TinyEncoder(torch.nn.Module):
    def __init__(self, n_embed: int, n_heads: int, n_layers: int, vocab_size: int, block_size: int):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embed)
        self.blocks = torch.nn.ModuleList([Block(n_embed, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = torch.nn.LayerNorm(n_embed)
        # self.lm_head = torch.nn.Linear(n_embed, vocab_size)  # Remove or comment out

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2, f"Input shape must be [B, T], got {x.shape}"
        T = x.size(1)
        x = self.token_embedding_table(x)
        x = x + self.position_embedding_table(torch.arange(T, device=x.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # x = self.lm_head(x)  # Remove this line
        return x  # Returns hidden states of shape [B, T, n_embed]


class PositionSpecificAttention(torch.nn.Module):
    def __init__(self, n_embed: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = n_embed // n_heads
        self.scale = self.head_size ** -0.5

        self.W_Q = torch.nn.Linear(n_embed, n_embed)
        self.W_K = torch.nn.Linear(n_embed, n_embed)
        self.W_V = torch.nn.Linear(n_embed, n_embed)
        self.dropout = torch.nn.Dropout(dropout)
        self.out_proj = torch.nn.Linear(n_embed, n_embed)

    def forward(self, Q, K, V, attn_mask):
        C = Q.size(-1)
        B = Q.size(0)
        L_Q = Q.size(1)
        L_K = K.size(1)
        
        Q = self.W_Q(Q).view(B, L_Q, self.n_heads, self.head_size).transpose(1, 2)
        K = self.W_K(K).view(B, L_K, self.n_heads, self.head_size).transpose(1, 2)
        V = self.W_V(V).view(B, L_K, self.n_heads, self.head_size).transpose(1, 2)

        # Compute scores
        scores = (Q @ K.transpose(-2, -1)) * self.scale  # Shape: [B, n_heads, L_Q, L_K]

        # Adjust attn_mask shape and add to scores
        attn_mask = attn_mask.unsqueeze(1)  # Shape: [B, 1, L_Q, L_K]
        scores += attn_mask

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        A = attn_weights @ V # B, n_heads, L_Q, head_size
        A = A.transpose(1, 2).contiguous().view(B, L_Q, C)
        return self.out_proj(A)

class TinyDecoder(torch.nn.Module):
    def __init__(self, n_embed: int, n_heads: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = torch.nn.Embedding(512, n_embed)
        self.attention = PositionSpecificAttention(n_embed, n_heads, dropout)
        self.ff = FeedForward(n_embed)
        self.ln1 = torch.nn.LayerNorm(n_embed)
        self.ln2 = torch.nn.LayerNorm(n_embed)
        self.final_layer = torch.nn.Linear(n_embed, vocab_size)

    def forward(self, h_tilde_X, input_ids, attn_mask):
        B, L = input_ids.size()
        positions = torch.arange(L + 1, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        # Construct H1
        h_tilde_X_exp = h_tilde_X.unsqueeze(1).expand(-1, L + 1, -1)
        pos_emb = self.position_embedding_table(positions)
        H1 = h_tilde_X_exp + pos_emb

        # Construct H2
        token_emb = self.token_embedding_table(input_ids) + pos_emb[:, 1:, :]
        H2 = torch.cat([h_tilde_X.unsqueeze(1), token_emb], dim=1)

        # Apply attention
        H1_norm = self.ln1(H1)
        H2_norm = self.ln1(H2)
        attn_out = self.attention(H1_norm, H2_norm, H2_norm, attn_mask)

        # Add & Norm
        x = H1 + attn_out
        x = self.ln2(x)
        x = x + self.ff(x)
        x = self.ln2(x)

        # Output layer
        logits = self.final_layer(x)
        return logits


def create_attention_mask(input_ids, h_tilde_X_pos=0):
    L = input_ids.size(1)
    total_length = L + 1  # Account for h_tilde_X position
    attn_mask = torch.full((total_length, total_length), float('-inf'), device=input_ids.device)
    
    for i in range(total_length):
        attn_mask[i, :] = 0.0  # Allow attending to all positions
        attn_mask[i, i] = float('-inf')  # Prevent self-attention

    # Ensure h_tilde_X cannot attend to itself
    attn_mask[h_tilde_X_pos, h_tilde_X_pos] = float('-inf')

    return attn_mask


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits, labels):
        # Exclude the first position (h_tilde_X)
        logits = logits[:, 1:, :].contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)  # Do not exclude any positions here

        return self.loss_fn(logits, labels)


def mask_input(input_ids, mask_token_id, special_token_ids, masking_ratio):
    """
    Masks input tokens according to the specified masking ratio.
    Args:
        input_ids (torch.Tensor): Tensor of shape [T] or [B, T].
    Returns:
        masked_input_ids (torch.Tensor): Masked input IDs.
        labels (torch.Tensor): Labels for computing the loss.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # [1, T]

    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    B, T = input_ids.size()
    num_mask = max(1, int(round(T * masking_ratio)))

    for b in range(B):
        candidate_indices = [
            i for i in range(T)
            if input_ids[b, i].item() not in special_token_ids
        ]
        if len(candidate_indices) == 0:
            continue  # No tokens to mask
        masked_indices = np.random.choice(candidate_indices, num_mask, replace=False)
        for idx in masked_indices:
            labels[b, idx] = input_ids[b, idx]
            masked_input_ids[b, idx] = mask_token_id

    return masked_input_ids.squeeze(0), labels.squeeze(0)


class RetroMAEModel(torch.nn.Module):
    def __init__(self, n_embed: int, n_heads: int, n_layers: int, vocab_size: int, block_size: int):
        super().__init__()
        self.encoder = TinyEncoder(n_embed, n_heads, n_layers, vocab_size, block_size)
        self.decoder = TinyDecoder(n_embed, n_heads, vocab_size)
        self.loss_fn = Loss()

    def forward(self, encoder_input_ids, decoder_input_ids, attn_mask, decoder_labels):
        # Encoder forward pass
        encoder_hidden_states = self.encoder(encoder_input_ids)  # Shape: [B, T, n_embed]
        h_tilde_X = encoder_hidden_states[:, 0, :]  # [CLS] token embedding, shape: [B, n_embed]

        # Decoder forward pass
        logits = self.decoder(h_tilde_X, decoder_input_ids, attn_mask)

        # Compute loss
        loss = self.loss_fn(logits, decoder_labels)
        return loss


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        Add <cls> and <sep> tokens to the input text.
        Add padding tokens to the end of the input text.
        """

        text = self.data[idx:idx + self.block_size - 2]
        text = f"{CLS}{text}{SEP}"
        encoded_data = tokenize(text, inv_vocab)

        if len(encoded_data) < self.block_size:
            encoded_data += [inv_vocab[PAD]] * (self.block_size - len(encoded_data))
        elif len(encoded_data) > self.block_size:
            encoded_data = encoded_data[:self.block_size]


        input_ids = torch.tensor(encoded_data, dtype=torch.long)

        # Mask encoder input
        encoder_input_ids, _ = mask_input(
            input_ids, inv_vocab[MASK], [inv_vocab[token] for token in SPECIAL_TOKENS], encoder_masking_ratio
        )

        # Mask decoder input
        decoder_input_ids, decoder_labels = mask_input(
            input_ids, inv_vocab[MASK], [inv_vocab[token] for token in SPECIAL_TOKENS], decoder_masking_ratio
        )

        attn_mask = create_attention_mask(decoder_input_ids)

        return encoder_input_ids, decoder_input_ids, attn_mask, decoder_labels


if __name__ == '__main__':

    def get_data():
        path = Path('./dataset/text8')
        with path.open() as f:
            data = f.read()
        return data

    data = get_data()
    vocab = {i: bytes([i]).decode('utf-8') for i in range(128)}

    vocab.update({
        i: token
        for i, token in enumerate(SPECIAL_TOKENS, start=len(vocab))
    })

    inv_vocab = {v: k for k, v in vocab.items()}

    n_embed = 64
    n_heads = 8
    n_layers = 6
    block_size = 128
    vocab_size = len(vocab)
    encoder_masking_ratio = 0.15
    decoder_masking_ratio = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model
    model = RetroMAEModel(n_embed, n_heads, n_layers, vocab_size, block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    split_ratio = 0.8

    train = Dataset(data[:int(len(data) * split_ratio)], block_size)
    val = Dataset(data[int(len(data) * split_ratio):], block_size)

    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=False, num_workers=4)

    # Forward pass
    model.to(device).train()
    for encoder_input_ids, decoder_input_ids, attn_mask, decoder_labels in train_loader:
        encoder_input_ids = encoder_input_ids.to(device)  # Shape: [batch_size, T]
        decoder_input_ids = decoder_input_ids.to(device)  # Shape: [batch_size, T]
        attn_mask = attn_mask.to(device)                  # Shape: [batch_size, L+1, L+1]
        decoder_labels = decoder_labels.to(device)        # Shape: [batch_size, T]

        optimizer.zero_grad()
        loss_value = model(encoder_input_ids, decoder_input_ids, attn_mask, decoder_labels)
        loss_value.backward()
        optimizer.step()

        print(f"Loss: {loss_value.item()}")
