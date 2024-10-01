# N-gram Language Model
https://web.stanford.edu/~jurafsky/slp3/3.pdf

## Intro
N-gram language model is a probabilistic model for predicting the next word in a
sequence of $N$ word context.

## Definition
Given a sequence of words $w_1, w_2, \ldots, w_n$, the probability of the next
word $w_{n+1}$ is given by the conditional probability:
$$
P(w_{n+1} | w_1, w_2, \ldots, w_n) = P(w_{n+1} | w_n, w_{n-1}, \ldots, w_{n-N+1})
$$
where $N$ is the order of the N-gram model.

In practice, we approximate the conditional probability using the maximum
likelihood estimation:
$$
P(w_{n+1} | w_n, w_{n-1}, \ldots, w_{n-N+1}) =
\frac{C(w_{n+1}, w_n, w_{n-1}, \ldots, w_{n-N+1}, w_{n+1})}{C(w_n, w_{n-1}, \ldots, w_{n-N+1})}
$$
where $C(\cdot)$ is the count of the words in the corpus.

**Example:**

Sentence: "What you cannot create, you do not understand."

$$
P(\text{understand} | \text{you}, \text{do}, \text{not}) = 
\frac{C(\text{you}, \text{do}, \text{not}, \text{understand})}{C(\text{you}, \text{do}, \text{not})}
$$

### Details
Probability of the sequence such as $P(w_1, w_2, \ldots, w_n)$ can be computed
using the *chain rule*:
$$
P(w_1 \ldots w_n) = P(w_1)P(w_2|w_1)P(w_3|w_1, w_2) \ldots P(w_n|w_1, \ldots, w_{n-1}) =
\prod_{i=1}^{n} P(w_i|w_1, \ldots, w_{i-1})
$$

The idea of N-gram model is to approximate the probability of the entire
sequence $w_n$ using the conditional probabilities of the last $N$ words
(e.g. [Bi-gram](#Bi-gram)).




## Special Cases
### Unigram
Unigram model is a special case of N-gram model where $N = 1$ and the
probability of the next word is independent of the previous words.

### Bi-gram


