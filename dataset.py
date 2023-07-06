import random

import matplotlib.pyplot as plt
import torch

def get_data(path = None):
    if not path:
        path = 'names.txt'

    with open(path, 'r') as f:
        words = f.read().splitlines()
    return words

def statistics(words):
    print('Total number of words: {}'.format(len(words)))
    print('Total number of unique words: {}'.format(len(set(words))))
    print('Average word length: {}'.format(sum(len(word) for word in words) / len(words)))
    print('Maximum word length: {}'.format(max(len(word) for word in words)))
    print('Minimum word length: {}'.format(min(len(word) for word in words)))

def build_dataset(words, s2i, block_size=3):
    """
    Create a training data X, Y, where each training example contain 3 previous
    characters (block_size).
    """
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = s2i[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)     # (N, 3)
    Y = torch.tensor(Y)     # (N)
    return X, Y


def visualize_embeddings(C, i2s):
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), i2s[i], ha='center', va='center', color='white')
    plt.grid('minor')
    plt.show()

def get_datasets(block_size=3):
    """Get the dataset and create a mapping from characters to integers.

    Returns:
        dataset: dictionary with keys 'train', 'val', 'test' and values
            (X, Y) where X is a tensor of shape (N, block_size) and Y is a
            tensor of shape (N).
        s2i: dictionary mapping each character to a unique integer
        i2s: dictionary mapping each integer back to its corresponding character
    """
    words = get_data()
    char_int_map = sorted(set(''.join(words)))
    s2i = {c: i+1 for i, c in enumerate(char_int_map)}
    s2i['.'] = 0
    i2s = {i: c for c, i in s2i.items()}

    # Training split, val split, test split
    # 80%, 10%, 10%

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    datasets = {
            'train': build_dataset(words[:n1], s2i, block_size=block_size),
            'val': build_dataset(words[n1:n2], s2i, block_size=block_size),
            'test': build_dataset(words[n2:], s2i, block_size=block_size), 
            }
    return datasets, s2i, i2s

