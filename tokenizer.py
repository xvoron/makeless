"""
Based on karpathy's new video about tokenization.
https://www.youtube.com/watch?v=zduSFxRajkE
"""


class Tokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}

    def train(self, text: str):
        data = list(text.encode('utf-8'))
        num_of_merges = self.vocab_size - 255

        val = 255

        for _ in range(num_of_merges):
            pairs = self.get_counts(data)
            top_pair = max(pairs, key=pairs.get)
            val += 1
            print(f"Merging {top_pair} -> {val}")
            data = self.merge(data, top_pair, val)
            self.merges[top_pair] = val

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx  in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

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
            if i < len(data) - 1 and (data[i], data[i + 1]) == pair:
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
        tokens = list(data.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self.get_counts(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens


test_string_str = "aaabbdaaabbbac"

tokenizer = Tokenizer(258)
tokenizer.train(test_string_str)
print(tokenizer.merges)
print(tokenizer.encode(test_string_str))
decoded = tokenizer.decode(tokenizer.encode(test_string_str))
print(decoded)
assert decoded == test_string_str
