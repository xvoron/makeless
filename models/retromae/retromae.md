# RetroMAE

MAE - masked auto-encoder.

Dense retrieval is a task of retrieving the most relevant documents from a large collection of
documents given a query. The task is usually formulated as a ranking problem, where the model is
trained to rank the relevant documents higher than the irrelevant ones.


> Auto-encoder come from computer vision, where the task of network is to learn to reconstruct the
> input corrupted by noise. The network is trained to minimize the difference between the noisy
> input and the clean output. The network is usually composed of encoder and decoder. The encoder is
> responsible for encoding the input into a latent representation, and the decoder is responsible
> for decoding the latent representation back to the input space.

MAE workflow is using the different masking strategies (asymmetric model structure) for the encoder
and decoder.

The encoder is BERT-like model, and the decoder is one-layer transformer.

Asymmetric masking ratios:
- encoder: $15~30\%$ masking.
- decoder: $50~70\%$ masking.

BERT with MLM and Seq2Seq has lower sentence-level representation capability. So they not so good at
retrieval tasks.

Ways to train the model for retrieval tasks:
- Self-contrastive learning: discriminate between the positive and negative samples. 
- Auto-encoding.

Two factors that are critical for auto-encoding:
1. The reconstruction task must be challenging.
2. Fully utilize the data.

# Architecture
Encoder is BERT-like model $\Phi_{enc}(\cdot)$ generate sentence embedding.
Decoder is one-layer transformer $\Phi_{dec}(\cdot)$ for sentence reconstruction.

$X$ is input sentence. $\tilde{X}_{enc}$ is the masked input sentence for the encoder.
The embedding of the input sentence encoded by the encoder is $\textbf{h}_{\tilde{X}}$.
$\tilde{X}_{dec}$ is the masked input sentence for the decoder. Together with
$\textbf{h}_{\tilde{X}}$, the original sentence $X$ is reconstructed by the decoder.

$$
\textbf{h}_{\tilde{X}} \leftarrow \Phi_{enc}(\tilde{X}_{enc})
$$

> BERT with 12 layers and 768 hidden units.
`[CLS]` token is used as the sentence embedding.

$$
\textbf{H}_{\tilde{X}_{dec}} \leftarrow
\[\textbf{h}_{\tilde{X}}, \textbf{e}_{x_1} + \textbf{p_1}, \dots, \textbf{e}_{x_n} + \textbf{p_n}\]
$$
where $\textbf{e}_{x_i}$ is the embedding of the $x_i$ token in the input sentence $X$,
to which an extra positional embedding $\textbf{p_i}$ is added.
$\Phi_{dec}$ is learning to reconstruct the original sentence $X$ by:

$$
L_{dec} = \sum_{x_i \in \text{masked}} \text{CE} (x_i | \Phi_{dec}(\textbf{H}_{\tilde{X}_{dec}}))
$$
where $\text{CE}$ is the cross-entropy loss.


# TODO:
- What is PQ and HNSW?
- Seq2Seq model
- T5 model
