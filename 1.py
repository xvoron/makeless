import torch
import matplotlib.pyplot as plt

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



def create_bigrams(words):
    # Bi-gram
    # b = {}
    # for w in words:
    #     chars = ['<S>'] + list(w) + ['<E>']
    #     for ch1, ch2 in zip(chars, chars[1:]):
    #         bigram = (ch1, ch2)
    #         b[bigram] = b.get(bigram, 0) + 1
    # sort_bigrams = sorted(b.items(), key= lambda kv: -kv[1])

    # 26 chars + <S> + <E> = 28
    # 28 * 28 = 784
    N = torch.zeros((27, 27), dtype=torch.int32)

    char_int_map = sorted(set(''.join(words)))

    s2i = {c: i+1 for i, c in enumerate(char_int_map)}
    s2i['.'] = 0
    i2s = {i: c for c, i in s2i.items()}

    # Rows represents probability of the row_i char followed by all chars
    # Columns represents probability that before col_i char is some char
    for w in words:
        chars = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            idx1 = s2i[ch1]
            idx2 = s2i[ch2]
            N[idx1, idx2] += 1
    return N


def visualize_bigrams(N):
    plt.imshow(N)
    plt.show()

if __name__ == '__main__':
    words = get_data()
    N = create_bigrams(words)


    # Deterministic torch generator
    g = torch.Generator().manual_seed(2147483647)

    """
    Add +1 for smoother model
    So there is no 0 in P; then if we will calculate loss; then there is no inf will
    be represented in loss function. log(0) = -inf.

    Model smoothing technique.
    """
    P = (N + 1).float()
    # Broadcasting
    # 27, 27
    # 27, 1
    # Be very patient on this vvvv
    # We need to normalize by row vectors so 1 here
    P /= P.sum(dim=1, keepdim=True)
    # In place operations are faster

    for _ in range(10):
        idx = 0
        out = []
        while True:
            p = P[idx]

            """
            Return samples according to the probability distribution
            [0.6, 0.2, 0.2]

            with num_samples = 5 will return something like

            [0, 1, 0, 2, 0]
            """

            idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(i2s[idx])
            if idx == 0:
                break
        print(''.join(out))


    """
    # Maximum Likelihood estimation

    Likelihood is the product of the given probabilities.

    When you train the model this product must be high as possible.

    Log Likelihood estimation is the sum of the given log probabilities.

    log(a*b*c) = log(a) + log(b) + log(c)

    Negative log_likelihood is just - log_likelihood
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
    print(f"Average NLL: {-log_likelihood/n}")


