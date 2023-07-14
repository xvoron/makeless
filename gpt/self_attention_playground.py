import torch
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(1337)

B, T, C = 2, 3, 4 

x = torch.randn(B, T, C)

head_size = 5

# The key is what do token contain
key = nn.Linear(C, head_size, bias=False)
# the query is what the toking is looking for
query = nn.Linear(C, head_size, bias=False)
# the value is what the token is going to provide to the further layers
value = nn.Linear(C, head_size, bias=False)

k = key(x)
q = query(x)
print(f"{k.shape} = {key.weight.shape} @ {x.shape}")
print(f"{q.shape} = {query.weight.shape} @ {x.shape}")

wei = q @ k.transpose(-2, -1) * head_size ** -0.5
print(f"{wei.shape} = {q.shape} @ {k.transpose(-2, -1).shape}")




# Masking the lower triangle
# That means that the model can only attend to the past
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)
print(f"{wei=}")

v = value(x)
out = wei @ v
print(f"{out.shape} = {wei.shape} @ {x.shape}")


