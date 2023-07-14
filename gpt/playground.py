import torch
from torch.nn import functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randint(0, 10, (B, T, C)).float()

# 1.0
# x[b, t] = mean_{i<=t} x[b, i]

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)
print(x[0])
print(xbow[0])


# Math hack of the upper
print("Hack")
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
print(f"{a=}")
b = torch.randint(0, 10, (3, 2)).float()
print(f"{b=}")
c = a @ b
print(f"{c=}")


# Using hack and  1.0
weights = torch.tril(torch.ones(T, T))
weights = weights / weights.sum(1, keepdim=True)
print(f"{weights=}")

# (T, T) @ (B, T, C) ---->   (B, T, T) @ (B, T, C)  ----> (B, T, C)
xbow2 = weights @ x 
print(torch.allclose(xbow, xbow2))


# Using softmax

tril = torch.tril(torch.ones(T, T))
weights = torch.zeros((T, T))
weights = weights.masked_fill(tril == 0, float('-inf'))
weights = F.softmax(weights, dim=-1)
xbow3 = weights @ x
print(torch.allclose(xbow, xbow3))



