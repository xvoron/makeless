# https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4
# https://jmlr.org/papers/volume3/tmp/bengio03a.pdf
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

from main import get_data

DEBUG = False

def print_(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def to_1hot(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()

def softmax(x):
    x = x - max(x)
    x = x.exp()
    return x / x.sum(1, keepdims=True)

def build_dataset(words, s2i):
    """
    Create a training data X, Y, where each training example contain 3 previous
    characters (block_size).
    """
    block_size = 3
    X, Y = [], []
    for w in words:
        print_(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = s2i[ch]
            X.append(context)
            Y.append(ix)
            print_(''.join(i2s[i] for i in context), '--->', i2s[ix])
            context = context[1:] + [ix]
    X = torch.tensor(X)     # (N, 3)
    Y = torch.tensor(Y)     # (N)
    return X, Y

def visualize_embeddings(C):
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), i2s[i], ha='center', va='center', color='white')
    plt.grid('minor')
    plt.show()




if __name__ == '__main__':
    words = get_data()
    char_int_map = sorted(set(''.join(words)))

    s2i = {c: i+1 for i, c in enumerate(char_int_map)}
    s2i['.'] = 0
    i2s = {i: c for c, i in s2i.items()}

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))


    block_size = 3
    # Training split, val split, test split
    # 80%, 10%, 10%
    Xtr, Ytr = build_dataset(words[:n1], s2i)
    Xval, Yval = build_dataset(words[n1:n2], s2i)
    Xtest, Ytest = build_dataset(words[n2:], s2i)

    """
    -------------------- PARAMETERS -------------------------
    """

    """
    # Look-up table C
    Starting with small look-up table, where each character is embedded using 2
    numbers for encoding.
    """
    C = torch.randn((27,10))
    W1 = torch.randn((30, 200))
    b1 = torch.randn(200)
    W2 = torch.randn((200, 27))
    b2 = torch.randn(27)
    parameters = [C, W1, b1, W2, b2]
    print_(f"Number of parameters: {sum(p.nelement() for p in parameters)}")



    # The same as C[X] X are (N, 3) tensor, where each of the 3 values are
    # characters in 0...27, N is number of examples.
    # emb = to_1hot(torch.tensor(X), num_classes=27) @ C  # (N, 3, 2)
    emb = C[Xtr]

    # torch.cat((emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]), dim=1).shape
    # equivalent for this will be
    # torch.cat(torch.unbind(emb, 1), 1)
    # Better way is to use following
    x = emb.view(-1, 30)     # (N, 6)
    h = torch.tanh(x @ W1 + b1)     # (N, 100)

    # (N, 100) @ (100, 27) = (N, 27)
    logits = h @ W2 + b2

    """
    # This is the same as cross entropy loss
    prob = softmax(logits)  # (N, 27)
    print_(f"{prob.shape=}")
    loss = -prob[torch.arange(prob.shape[0]), Y].log().mean()
    print_(f"{loss=}")
    """

    # Cross entropy loss
    loss = F.cross_entropy(logits, Ytr)
    print_(f"{loss=}")


    for p in parameters:
        p.requires_grad = True

    lre = torch.linspace(-3, 0, 1000)
    lrs = 10**lre
    # plt.plot(lrs)
    # plt.show()


    losses = []
    epochs = 50000
    for e in tqdm(range(epochs)):
        # Mini-batch
        ixs = torch.randint(0, Xtr.shape[0], (32,))
        # Forward pass
        emb = C[Xtr[ixs]]
        x = emb.view(-1, 30)     # (N, 6)
        h = torch.tanh(x @ W1 + b1)     # (N, 100)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ixs])

        # Backward pass
        for p in parameters:
            p.grad = None

        loss.backward()

        lr = .1
        for p in parameters:
            p.data += - lr * p.grad
        losses.append(loss.log10().item())

    print(f"Train Loss {loss.item()}")
    # plt.plot(losses)
    # plt.show()

    # plt.plot(lrs, losses)
    # plt.show()

    # Test
    emb = C[Xval]
    x = emb.view(-1, 30)
    h = torch.tanh(x @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yval)
    print(f"Val loss: {loss}")

    # visualize_embeddings(C)


    # Sampling from model

    for _ in range(20):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])] # (1, block_size, d)
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(i2s[i] for i in out))




