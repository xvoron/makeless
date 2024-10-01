from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import torch


def lev(a, b):
    if not a: return len(b)
    if not b: return len(a)
    if a[0] == b[0]: return lev(a[1:], b[1:])
    return 1 + min(lev(a[1:], b), lev(a, b[1:]), lev(a[1:], b[1:]))


def plot_embedding(words, embedding, word_to_ix, top=150):
    """
    words: raw tokens
    embedding: torch.nn.Embedding() object
    """
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

def find_similar(word, words, embedding, word_to_ix, n=5, from_total=5000):
    distance = []
    with torch.no_grad():
        y = embedding(word_to_ix[word]).numpy().reshape(1, -1)
        total = Counter(words).most_common(from_total)
        for w, _ in total:
            x = embedding(word_to_ix[w]).numpy().reshape(1, -1)
            distance.append(cosine_distances(x, y)[0][0])
    
    distance = np.array(distance)
    top_n = distance.argsort()[1:n+1]
    
    return [total[ix][0] for ix in top_n]
