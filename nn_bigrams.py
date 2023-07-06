from main import get_data
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_bigrams(words):
    # Bi-gram

    char_int_map = sorted(set(''.join(words)))

    s2i = {c: i+1 for i, c in enumerate(char_int_map)}
    s2i['.'] = 0
    i2s = {i: c for c, i in s2i.items()}

    # Rows represents probability of the row_i char followed by all chars
    # Columns represents probability that before col_i char is some char
    xs, ys = [], []
    for w in words:
        chars = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            idx1 = s2i[ch1]
            idx2 = s2i[ch2]
            xs.append(idx1)
            ys.append(idx2)
    return torch.tensor(xs), torch.tensor(ys), s2i, i2s

def to_1hot(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()

def softmax(x):
    x = x.exp()
    return x / x.sum(1, keepdims=True)


if __name__ == '__main__':
    words = get_data()
    xs, ys, s2i, i2s = create_bigrams(words)

    xs1h = to_1hot(xs, 27)
    ys1h = to_1hot(ys, 27)

    nin = xs1h.shape[1]
    n_neurons = 27
    W = torch.randn((nin, n_neurons), requires_grad=True)


    epoch = 100
    losses = []

    for e in tqdm(range(epoch)):
        # Forward pass
        logits = xs1h @ W   # log counts
        # Logits return row with weights of the current position of one hot vector
        """
        Given output y will give us log(counts);
        so we will exp this output of y: e^y.

        This is softmax function other way around.
        """
        counts = logits.exp()   # equivalent to N
        probs = counts / counts.sum(1, keepdims=True)
        regularization = (W**2).mean()
        loss = -probs[torch.arange(xs1h.shape[0]), ys].log().mean() + 0.01*regularization
        losses.append(loss.item())

        # Backward pass
        W.grad = None

        loss.backward()

        # W.grad is influence of weight on a loss.
        W.data -= 50 * W.grad

    print(f'Final Loss: {loss}')
    # plt.plot(losses)
    # plt.show()


    # Infer the model

    for i in range(5):
        out = []
        ix = 0

        while True:
            x1h = to_1hot(torch.tensor([ix]), 27)
            logits = x1h @ W
            pred = softmax(logits)

            ix = torch.multinomial(pred, num_samples=1, replacement=True).item()
            out.append(i2s[ix])
            if ix == 0:
                break
        print(''.join(out))

