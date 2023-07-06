import torch

from torchlike import BatchNorm1d, Linear, Tanh
from dataset import get_datasets

# Hyperparameters

n_embd = 10
n_hidden = 100
vocab_size = 27

# Number of previous characters
block_size = 3

# Dataset
datasets, s2i, i2s = get_datasets(block_size)


# Lookup table with embeddings for each character
C = torch.randn((vocab_size, n_embd))

# Bias is unnecessary because of batch normalization
layers = [
        Linear(block_size * n_embd, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
    ]


# If there is no batch normalization than the model must be initialized with
# respect to all the layers and activations to have good variance and be robust.
# Using batch normalization negates the need for this.
with torch.no_grad():
    # make less confident predictions of the last layers
    # to make model more robust
    # The last layer now batch norm
    layers[-1].gamma *= 0.1

    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            # apply gain from 
            layer.weight *= 1 # 5 / 3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True


# Training loop

max_steps = 200_00
batch_size = 32
loss_i = []

# Show how much the parameters are updated relative to the parameter value
update_to_data_ratio = []

Xtr, Ytr = datasets['train']

for i in range(max_steps):
    # mini-batch construction
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)  # flatten

    # Forward pass
    for layer in layers:
        x = layer(x)
    loss = torch.nn.functional.cross_entropy(x, Yb)

    # Backward pass

    for p in parameters:
        p.grad = None

    loss.backward()

    # Update parameters
    lr = 0.1 if i < 100_00 else 0.01
    for p in parameters:
        p.data -= torch.tensor(lr) * p.grad


    if not i % 1000:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
        loss_i.append(loss.item())

        with torch.no_grad():
            update_to_data_ratio.append(
                    [(lr * p.grad.std() / p.data.std()).log().item() for p in parameters]
            )

if True:
    import matplotlib.pyplot as plt
    # Plotting
    plt.figure()
    plt.plot(loss_i)

    plt.figure()
    plt.plot(update_to_data_ratio)
    plt.plot([0, len(update_to_data_ratio)], [-3, -3], 'k--')
    plt.show()
