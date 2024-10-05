# BERT
- lstm Bidirectional 
- gpt paper https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- ELMo concatenation of left-to-right and right-to-left LSTM

**BERT (Bidirectional Encoder Representations from Transformer)**

- CLS token


Architecture:
Multi-layer bidirectional Transformer encoder.

$L$ - number of layers.
$H$ - hidden size.
$A$ - number of self-attention heads.

Base: $L=12$, $H=768$, $A=12$.
Large: $L=24$, $H=1024$, $A=16$.

WordPiece tokenization (30k tokens).
CLS token is first token of every sequence.
Aggregate sequence representation for classification tasks.

> Sequence is a input sequence of tokens (sentence, paragraph, document).

1. [SEP] - separator token for pair of sentences (one sequence).
2. Add learned embeddings to token embeddings to indicate sentence A or B.
> $E_{A} + e_{token} = E_{token}$

Input embedding $E$.
Output final hidden vector:
`[CLS]` = $C \in \mathbb{R}^{H}$
Hidden vector for the input token $T_i \in \mathbb{R}^{H}$.

## Pre-training

### Masked Language Model (MLM)
- Mask some tokens from input and objective is to predict the original masked tokens based on the context.

Mask 15% of tokens for MLM.

Contrast with Denoising Autoencoder (DAE) that try to reconstruct the original input from the
corrupted input. Here only predict the masked tokens.

To avoid mismatch between pre-training (where `[MASK]` token is used and fine-tuning where 
there is no `[MASK]` token the researchers do the following:
Chose 15% of tokens at random.
- 80% of the time replace with `[MASK]`.
- 10% of the time replace with random token.
- 10% of the time keep the same.

Then $T_i$ is used to predict the original token with Cross-Entropy loss.

### Next Sentence Prediction (NSP) 
Chose two sentences A and B. 50% B is the next sentence `[IsNext]`.
50% B is a random sentence `[NotNext]`.
