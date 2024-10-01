from collections import Counter
from typing import Generator

import numpy as np
import torch
import torch.nn as nn
import random
from base import Dataset, PAD, UNK, get_corpus, get_vocab, cut_corpus
from torch.utils.tensorboard import SummaryWriter
from utils import plot_embedding


def cos_sim(x, y):
    return np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y))

def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
    """
    
    # Here we're calculating the cosine similarity between some random words and 
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.
    
    # sim = (a . b) / |a||b|
    
    embed_vectors = embedding.weight
    
    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
        
    return valid_examples, similarities


def subsample(words: list[int], threshold: float =1e-5) -> list[int]:
    """
    In original paper implementation is different:
        P(w_i) = (sqrt(z(w_i)/0.001) + 1) * (0.001/z(w_i))

    """
    words_counts = Counter(words)
    freqs = get_freqs(words)
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in words_counts}

    # 1 - p_drop is the probability of keeping the word
    # so if np.random.random() = 0.8 and p_drop = 0.5 (1 - 0.5 = 0.5) we keep the word
    return [word for word in words if np.random.random() < (1 - p_drop[word])]

def get_freqs(words: list[int]) -> dict:
    total_words = len(words)
    words_counts = Counter(words)
    return {word: count/total_words for word, count in words_counts.items()}

def get_target(words, idx, window_size=5):
    """
    words = [0, 1, 2, 3, 4, 5, 6]
    idx = 2
    window_size = 2

    R = 1 -> return [1, 3]
    R = 2 -> return [0, 1, 3, 4]

    So the output is always symmetric around the index, except the index itself
    is not included and beginning and the end of the list are edge cases.

    """
    R = np.random.randint(1, window_size+1)
    start = idx - R if idx - R > 0 else 0
    stop = idx + R
    return words[start:idx] + words[idx+1:stop+1]

def get_data(words, window_size: int = 5) -> tuple[list[int], list[int]]:
    X, Y = [], []
    for idx in range(len(words)):
        x = words[idx]
        y = get_target(words, idx, window_size)
        X.extend([x] * len(y))
        Y.extend(y)
    return X, Y

def get_batches(words: list[int], batch_size: int = 32, window_size: int = 5) -> Generator:
    n_batches = len(words) // batch_size
    words = words[:n_batches * batch_size]  # drop last

    # all the batches than has different length than batch size ???
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            for by in batch_y:
                x.append(batch_x)
                y.append(by)
                if len(x) == batch_size:
                    yield x, y
                    x, y = x[batch_size:], y[batch_size:]

        # if batch looks like [0, 1, 2, 3, 4, ...] then
        # possible x, y pairs may look like:
        # x = [0, 0, 0, 1, 1, 2, 2, 2, 3, 4, ...]
        # y = [1, 2, 3, 0, 2, 0, 1, 3, 0, 0, ...]


class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim, noise_dist=None):
        super().__init__()
        self.noise_dist = noise_dist
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target: torch.Tensor, context: torch.Tensor, noise_words: torch.Tensor):
        """
        Note:
            B - batch size
            N - number of noise words
            e - embedding dimension
        """
        # (B) -> (B, e)
        target_embedding = self.in_embed(target)
        # (B) -> (B, e)
        context_embedding = self.out_embed(context)

        # (B * N) -> (B * N, e)
        noise_embedding = self.out_embed(noise_words)

        B, E = target_embedding.shape
        target_embedding = target_embedding.view(B, E, 1)
        context_embedding = context_embedding.view(B, 1, E)

        # (B, 1, e) * (B, e, 1) -> (B, 1, 1)
        context_loss = torch.bmm(context_embedding, target_embedding)
        # (B, 1, 1) -> (B, 1)
        context_loss = context_loss.sigmoid().log().squeeze()

        # (B * N, e) -> (B, N, e)
        noise_embedding = noise_embedding.view(B, -1, E)
        # (B, N, e) * (B, e, 1) -> (B, N, 1)
        noise_loss = torch.bmm(noise_embedding.neg(), target_embedding).sigmoid().log()
        # (B, N, 1) -> (B, N)
        noise_loss = noise_loss.squeeze().sum(1)

        loss = -(context_loss + noise_loss).mean()
        if loss.isnan():
            print("NAN")
            print(context_loss, noise_loss)
            exit()
        return loss

    def get_embedding(self, input_words):
        return self.in_embed(input_words)


if __name__ == "__main__":
    import datetime

    corpus = get_corpus()
    vocab, stoi, itos = get_vocab(corpus, 10000)
    words = cut_corpus(corpus, vocab)

    print(f"Total words: {len(words)}")
    print(f"Unique words: {len(set(words))}")

    int_words = [stoi[word] for word in words]

    # Sub-sampling
    sampled_int_words = subsample(int_words, 1e-4)
    print(f"Total words after sub-sampling: {len(sampled_int_words)}")

    embedding_dim = 100
    model = SkipGramNegativeSampling(len(vocab), embedding_dim)

    freqs = get_freqs(sampled_int_words)
    word_freqs = np.array(sorted(freqs.values(), reverse=True))
    unigram_dist = word_freqs / np.sum(word_freqs)
    noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    batch_size = 128
    num_samples = 5
    steps = 0
    print_every = 1000
    sw = SummaryWriter(f"runs/word2vec_{datetime.datetime.now().isoformat()}")

    # losses = []
    # for ii, (input_words, target_words) in enumerate(get_batches(sampled_int_words, batch_size=batch_size)):
    #     with torch.no_grad():
    #         input_words = torch.LongTensor(input_words)
    #         target_words = torch.LongTensor(target_words)

    #         noise_words = torch.multinomial(noise_dist, batch_size * num_samples, replacement=True)

    #         loss = model(input_words, target_words, noise_words)
    #         losses.append(loss.item())
    # print(f"Loss before training: {np.mean(losses)}")


    step = 0
    losses = []
    for e in range(epochs):
        for ii, (input_words, target_words) in enumerate(get_batches(sampled_int_words, batch_size=batch_size)):
            step += 1

            optimizer.zero_grad()
            input_words = torch.LongTensor(input_words)
            target_words = torch.LongTensor(target_words)

            noise_words = torch.multinomial(noise_dist, batch_size * num_samples, replacement=True)

            loss = model(input_words, target_words, noise_words)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())


            if step % print_every == 0:
                print("Epoch: {}/{}".format(e+1, epochs))
                print("Loss: ", loss.item()) # avg batch loss at this point in training
                valid_examples, valid_similarities = cosine_similarity(model.in_embed)
                _, closest_idxs = valid_similarities.topk(6)

                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [itos[int(idx.item())] for idx in closest_idxs[ii]][1:]
                    print(itos[int(valid_idx.item())] + " | " + ', '.join(closest_words))
                print("...\n")

            sw.add_scalar("Loss/train", np.mean(losses), step)
            losses = []


    fig = plot_embedding(words, model.in_embed, stoi)
    fig.savefig("word2vec.png")
    fig.show()
