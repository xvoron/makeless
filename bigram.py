from typing import Optional
import torch
import matplotlib.pyplot as plt
from utils import get_data

class BiGram:
    def __init__(self, tokens: int):
        self.tokens = tokens
        self.N = torch.zeros((tokens, tokens))
        self.s2i = {}
        self.i2s = {}

    @property
    def P(self):
        return self.N

    def create_bigrams(self, words: list):
        char_int_map = sorted(set(''.join(words)))

        self.s2i = {c: i+1 for i, c in enumerate(char_int_map)}
        self.s2i['.'] = 0
        self.i2s = {i: c for c, i in self.s2i.items()}

        # Rows represents probability of the row_i char followed by all chars
        # Columns represents probability that before col_i char is some char
        for w in words:
            chars = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chars, chars[1:]):
                idx1 = self.s2i[ch1]
                idx2 = self.s2i[ch2]
                self.N[idx1, idx2] += 1
        self._smooth(1)
        self._normalize()


    def visualize_bigrams(self, N: Optional[torch.Tensor] = None):
        if N is None:
            N = self.N
        plt.imshow(N)
        plt.show()

    def _smooth(self, n: int = 1):
        """Model smoothing technique.
        Add +n for smoother model
        So there is no 0 in P; then if we will calculate loss; then there is no inf will
        be represented in loss function. log(0) = -inf.
        """
        self.N += n

    def _normalize(self):
        self.N /= self.N.sum(dim=1, keepdim=True)

    def inference(self, num_examples: int = 1):
        ret = []

        for _ in range(num_examples):
            idx = 0
            out = []
            while True:
                p = self.P[idx]    # Get row of probabilities
                idx = torch.multinomial(p, num_samples=1, replacement=True).item()
                out.append(self.i2s[idx])   # Get char by index
                if idx == 0:    # Where index 0 is '.' (special start/end token)
                    break
            ret.append(''.join(out))
        return ret

def negative_log_likelihood(P: torch.Tensor, words: list, s2i: dict):
    """Negative Log Likelihood estimation.
    Likelihood is the product of the given probabilities.
    When you train the model this product must be high as possible.
    Log Likelihood estimation is the sum of the given log probabilities.
    log(a*b*c) = log(a) + log(b) + log(c)
    Negative log_likelihood is just - log_likelihood

    Return average negative log likelihood across all words.
    """
    log_likelihood = .0
    n = 0
    for w in words:
        chars = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            idx1 = s2i[ch1]
            idx2 = s2i[ch2]
            prob = P[idx1, idx2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1
            # print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")
    return -log_likelihood/n


if __name__ == '__main__':
    words = get_data()
    bigram = BiGram(tokens=27)
    bigram.create_bigrams(words)
    print('Negative log likelihood:', negative_log_likelihood(bigram.P, words, bigram.s2i))
    print('Inference:', bigram.inference(num_examples=10))


