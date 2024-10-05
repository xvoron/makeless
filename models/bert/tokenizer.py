from collections import defaultdict


class BPETokenizer:
    def __init__(self, special_tokens=None, vocab_size=50):
        """
        Initialize the BPE Tokenizer.

        :param special_tokens: A list of special tokens, e.g., ['[cls]', '[sep]'].
        :param vocab_size: The maximum number of tokens in the vocabulary.
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab = {}
        self.merges = []

    def get_vocab(self):
        """
        Get the current vocabulary.

        :return: The vocabulary dictionary.
        """
        return self.vocab

    def train(self, corpus):
        """
        Train the BPE Tokenizer on the provided corpus.

        :param corpus: A list of sentences (strings).
        """
        # Initialize token vocabulary with special tokens
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}

        # Tokenize sentences at the character level
        tokens = []
        for sentence in corpus:
            tokens.append(list(sentence) + ['</w>'])

        # Create pairs of characters to count their frequencies
        pair_freq = self.get_pair_frequencies(tokens)

        # Perform BPE token merges until vocab size limit is reached
        while len(self.vocab) < self.vocab_size:
            if not pair_freq:
                break

            # Find the most frequent pair
            most_frequent = max(pair_freq, key=pair_freq.get)
            if pair_freq[most_frequent] == 0:
                break

            # Merge the most frequent pair in all tokens
            tokens = self.merge_pair(most_frequent, tokens)

            # Save the merge to apply it later during tokenization
            self.merges.append(most_frequent)

            # Update pair frequencies
            pair_freq = self.get_pair_frequencies(tokens)

            # Add the merged pair to the vocabulary
            self.vocab[''.join(most_frequent)] = len(self.vocab)

    def get_pair_frequencies(self, tokens):
        """
        Get the frequency of adjacent token pairs in the tokenized corpus.

        :param tokens: A list of tokenized sentences.
        :return: A dictionary with token pairs as keys and their frequencies as values.
        """
        pair_freq = defaultdict(int)
        for token_list in tokens:
            for i in range(len(token_list) - 1):
                pair_freq[(token_list[i], token_list[i + 1])] += 1
        return pair_freq

    def merge_pair(self, pair, tokens):
        """
        Merge a specific pair of tokens in the tokenized corpus.

        :param pair: A tuple of two tokens to be merged.
        :param tokens: A list of tokenized sentences.
        :return: The tokenized sentences with the pair merged.
        """
        new_tokens = []
        bigram = ''.join(pair)
        for token_list in tokens:
            new_token_list = []
            i = 0
            while i < len(token_list):
                # Merge the pair
                if i < len(token_list) - 1 and (token_list[i], token_list[i + 1]) == pair:
                    new_token_list.append(bigram)
                    i += 2
                else:
                    new_token_list.append(token_list[i])
                    i += 1
            new_tokens.append(new_token_list)
        return new_tokens

    def tokenize(self, text):
        """
        Tokenize a sentence using the trained BPE model.

        :param text: The input text.
        :return: A list of tokens.
        """
        tokens = list(text) + ['</w>']

        # Apply BPE merges learned during training
        for merge in self.merges:
            tokens = self.merge_pair(merge, [tokens])[0]

        return tokens

# Example usage
corpus = [
    "<cls>",
    "<cls>",
    "<cls>",
    "hello world",
    "hello there",
    "hi world"
]

special_tokens = ['<cls>', '<sep>']
tokenizer = BPETokenizer(special_tokens=special_tokens, vocab_size=10)
tokenizer.train(corpus)

# Tokenize a sentence
sentence = "<cls> hello world"
tokenized_sentence = tokenizer.tokenize(sentence)
print("Tokenized sentence:", tokenized_sentence)
print("Vocabulary:", tokenizer.get_vocab())



class Tokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str] = []):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.vocab.update({
            idx: special_tokens[idx - 256].encode('utf-8')
            for idx in range(256, len(special_tokens) + 256)
        })
        self.special_tokens = {
            len(self.vocab)+ self.vocab_size + idx: special_token
            for idx, special_token in enumerate(special_tokens)
        }

    def train(self, text: str):
        assert self.vocab_size > 256
        num_merges = self.vocab_size - 256

        data: list[int] = list(text.encode('utf-8'))

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for idx in range(len(vocab), num_merges):
            stats = self.get_counts(data)
            top_pair = max(stats, key=stats.get)
            data = self.merge(data, top_pair, idx)

            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

        self.merges = merges
        self.vocab = vocab


    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode('utf-8')
        return vocab

    @staticmethod
    def get_counts(data: list[int]):
        counts = {}
        for pair in zip(data, data[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(data, pair, new_idx):
        _data = []
        i = 0
        while i < len(data):
            if data[i] == pair[0] and i < len(data) - 1 and data[i + 1 == pair[1]]:
                _data.append(new_idx)
                i += 2
            else:
                _data.append(data[i])
                i += 1
        return _data

    def decode(self, data):
        tokens = b''.join(self.vocab[idx] for idx in data)
        return tokens.decode("utf-8", errors="replace")

    def encode(self, data):
        tokens: list[int] = list(data.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self.get_counts(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

if __name__ == '__main__':
    tokenizer = Tokenizer(300, special_tokens=SPEC)
    print(tokenizer.vocab)
    print(tokenizer.encode(CLS))
    print(tokenizer.decode(tokenizer.encode(CLS)))

    string = "aaabbdaaabbbac"
    tokenizer.train(string)

    print(tokenizer.vocab)
    print(tokenizer.encode(string))
    print(tokenizer.decode(tokenizer.encode(string)))

