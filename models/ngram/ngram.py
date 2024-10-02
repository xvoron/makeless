from __future__ import annotations
from collections import Counter
import re

import polars as pl
import torch
import tqdm


UNK = '<unk>'
PAD = '<pad>'

STOI = dict[str, int]
ITOS = dict[int, str]


def text_preprocessor(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    return re.sub(r'\s+', ' ', text)


def get_corpus() -> str:
    """Get the corpus of the data as a string.

    Note:
        Use only the `rss_title` column.
    """
    data = pl.read_csv('./dataset/sportoclanky.csv')
    data = data.select([
        data['rss_title'],
    ])
    corpus = []
    for row in data.iter_rows():
        corpus.append(row[0])

    return text_preprocessor(' '.join(corpus))


def get_vocab(corpus: str, vocab_size: int = 10000) -> tuple[list[str], STOI, ITOS]:
    counts =  Counter(corpus.split())
    vocab = [word for word, _ in counts.most_common(vocab_size - 2)]
    vocab = list(set(vocab)) + [PAD, UNK]
    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for word, i in stoi.items()}
    return vocab, stoi, itos


def cut_corpus(corpus: str, vocab: list[str]) -> list[str]:
    return [word if word in vocab else UNK for word in corpus.split()]


class Ngram:
    def __init__(self, n: int, vocab: list[str], stoi: STOI, itos: ITOS):
        """Initialize the n-gram model.

        Args:
            n (int): n-gram order
            vocab (list[str]): vocabulary of the model
        """
        self.n = n
        self.vocab = vocab
        self.stoi = stoi
        self.itos = itos

        # n-gram n-dimensional tensor
        self.ngrams = torch.zeros([len(vocab)] * n)

    def train(self, text: list[str]):
        # There is no training for n-gram model in the traditional ml sense
        # We just count the n-grams in the text and normalize them
        total = len(text) - self.n + 1
        for i in tqdm.tqdm(range(total), total=total):
            indices = [self.stoi[text[j]] for j in range(i, i + self.n)]
            self.ngrams[tuple(indices)] += 1

        self.ngrams += 1
        # Probably there is no need in clamp here because the smoothing is already applied by adding
        # 1 to all counts in the previous line
        self.ngrams /= self.ngrams.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return self

    def get_features(self) -> torch.Tensor:
        return self.ngrams

    @classmethod
    def from_ngram(cls, ngram: Ngram):
        return cls(ngram.n, ngram.vocab, ngram.stoi, ngram.itos)


class NgramGenerator(Ngram):
    def generate(self, sentence: list[str], length: int = 10) -> str:
        if len(sentence) < self.n - 1:
            sentence = [PAD] * (self.n - 1 - len(sentence)) + sentence

        for _ in range(length):
            context = [self.stoi.get(word, self.stoi[UNK]) for word in sentence[-self.n+1:]]
            next_word_probs = self.ngrams[tuple(context)]
            # Sample next word from the distribution
            next_word_idx = torch.multinomial(next_word_probs, 1).item()
            sentence.append(self.itos[int(next_word_idx)])

        return ' '.join(sentence)


class NgramCorrector(Ngram):
    def correct_spelling(self, context: list[str], word: str) -> str:
        if len(context) < self.n - 1:
            context = [PAD] * (self.n - 1 - len(context)) + context

        context = context[-self.n+1:]

        context_idx = [self.stoi.get(word, self.stoi[UNK]) for word in context]
        all_probs = self.ngrams[tuple(context_idx)]
        filtered = [idx for idx, prob in enumerate(all_probs) if prob > all_probs.mean() - all_probs.std()]
        if not filtered:
            print(f"No similar words found for {word} in context {context}")
            return word
        top = sorted(filtered, key=lambda x: lev(word, self.itos[x]))[0]
        return self.itos[top]

    def process_query(self, query: list[str]) -> float:
        query_idx = [self.stoi.get(word, self.stoi[UNK]) for word in query]
        probs = self.ngrams[query_idx]
        return probs.mean().item()



def lev(a, b) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    if a[0] == b[0]:
        return lev(a[1:], b[1:])
    return 1 + min(lev(a[1:], b), lev(a, b[1:]), lev(a[1:], b[1:]))


class Index:
    def __init__(self):
        self.index = {}
        self.vocab = [PAD, UNK]
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.itos = {i: word for i, word in enumerate(self.vocab)}

    def add_document(self, document: str):
        text = text_preprocessor(document).split()
        self.update_vocab(text)
        self.index[document] = None

    def update_vocab(self, text: list[str]):
        self.vocab = list(set(self.vocab + text))
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.itos = {i: word for i, word in enumerate(self.vocab)}

    def build(self):
        for doc, _ in self.index.items():
            self.index[doc] = Ngram(1, self.vocab, self.stoi, self.itos).train(text_preprocessor(doc).split()).get_features()

    def search(self, query: str):
        q = text_preprocessor(query).split()
        q = [self.stoi.get(word, self.stoi[UNK]) for word in q]
        results = []
        for doc, features in self.index.items():
            probs = features[q]
            results.append((doc, probs.mean().item()))
        return sorted(results, key=lambda x: x[1], reverse=True)[0]


if __name__ == "__main__":

    corpus = get_corpus()
    vocab, stoi, itos = get_vocab(corpus, 2000)
    words = cut_corpus(corpus, vocab)


    ngram_model = Ngram(3, vocab, stoi, itos)
    ngram_model.train(words)

    # Krčmář dojel v hromadném závodě devátý

    context = ['krčmář', 'dojel', 'v']
    target = 'hromadem'
    print(f"Text generation for 'krčmář dojel v' : {NgramGenerator.from_ngram(ngram_model).generate(context.copy())}")
    print(f"Correct spelling for {target}: {NgramCorrector.from_ngram(ngram_model).correct_spelling(context.copy(), target)}")


    doc1_content = "The quick brown fox jumps over the lazy dog"
    doc2_content = "The quick brown fox jumps over the quick dog"

    index = Index()
    index.add_document(doc1_content)
    index.add_document(doc2_content)
    index.build()
    print(f"Search results for 'the quick dog': {index.search('the quick dog')}")
