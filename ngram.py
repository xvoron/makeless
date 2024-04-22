import torch

from base import cut_corpus, get_corpus, get_vocab, preprocess
from utils import lev


class Ngram:
    def __init__(self, n: int, vocab: list[str]):
        self.n = n
        self.vocab = vocab
        self.stoi = {word: i for i, word in enumerate(vocab)}
        self.itos = {i: word for i, word in enumerate(vocab)}
        self.ngrams = torch.zeros([len(vocab)] * n)

    def train(self, text: list[str]):
        for i in range(len(text) - self.n + 1):
            indeces = [self.stoi[text[j]] for j in range(i, i + self.n)]
            self.ngrams[tuple(indeces)] += 1

        self.ngrams += 1
        self.ngrams /= self.ngrams.sum(-1, keepdim=True)
        return self

    def generate(self, sentence: list[str], length: int = 10) -> str:
        if len(sentence) < self.n - 1:
            sentence = ['<PAD>'] * (self.n - 1 - len(sentence)) + sentence

        for _ in range(length):
            context = [self.stoi[word] for word in sentence[-self.n+1:]]
            next_word_probs = self.ngrams[tuple(context)]
            next_word_idx = torch.multinomial(next_word_probs, 1).item()
            sentence.append(self.itos[int(next_word_idx)])

        return ' '.join(sentence)

    def correct_spelling(self, context: list[str], word: str) -> str:
        if len(context) < self.n - 1:
            context = ['<PAD>'] * (self.n - 1 - len(context)) + context

        context = context[-self.n+1:]

        context_idx = [self.stoi[word] for word in context]
        all_probs = self.ngrams[tuple(context_idx)]
        filtered = [idx for idx, prob in enumerate(all_probs) if prob > all_probs.mean() - all_probs.std()]
        if not filtered:
            print(f"No similar words found for {word} in context {context}")
            return word
        top = sorted(filtered, key=lambda x: lev(word, self.itos[x]))[0]
        return self.itos[top]

    def process_query(self, query: list[str]) -> float:
        query_idx = [self.stoi[word] for word in query]
        probs = self.ngrams[query_idx]
        return probs.mean().item()

    def get_features(self) -> torch.Tensor:
        return self.ngrams


class Index:
    def __init__(self):
        self.index = {}
        self.vocab = ['<PAD>', '<UNK>']
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.itos = {i: word for i, word in enumerate(self.vocab)}

    def add_document(self, document: str):
        text = preprocess(document).split()
        self.update_vocab(text)
        self.index[document] = None

    def update_vocab(self, text: list[str]):
        self.vocab = list(set(self.vocab + text))
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.itos = {i: word for i, word in enumerate(self.vocab)}

    def build(self):
        for doc, _ in self.index.items():
            self.index[doc] = Ngram(1, self.vocab).train(preprocess(doc).split()).get_features()

    def search(self, query: str):
        q = preprocess(query).split()
        q = [self.stoi.get(word, self.stoi['<UNK>']) for word in q]
        results = []
        for doc, features in self.index.items():
            probs = features[q]
            results.append((doc, probs.mean().item()))
        return sorted(results, key=lambda x: x[1], reverse=True)[0]


if __name__ == "__main__":

    corpus = get_corpus()[:10000]
    vocab, stoi, itos = get_vocab(corpus, 1000)
    words = cut_corpus(corpus, vocab)


    ngram_model = Ngram(3, vocab)
    ngram_model.train(words)

    # Krčmář dojel v hromadném závodě devátý

    context = ['krčmář', 'dojel', 'v']
    target = 'hromadem'
    print(f"Text generation for 'krčmář dojel v' : {ngram_model.generate(context.copy())}")
    print(f"Correct spelling for {target}: {ngram_model.correct_spelling(context.copy(), target)}")


    doc1_content = "The quick brown fox jumps over the lazy dog"
    doc2_content = "The quick brown fox jumps over the quick dog"

    index = Index()
    index.add_document(doc1_content)
    index.add_document(doc2_content)
    index.build()
    print(f"Search results for 'the quick dog': {index.search('the quick dog')}")
