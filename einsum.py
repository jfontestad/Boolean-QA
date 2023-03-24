import numpy as np

# u = np.full((2, 3), 2)
#
# print(u)
# print(np.einsum("ij->j", u))
# print(np.einsum("ij->i", u))

import torch

x = torch.rand(4, 4)
y0 = torch.rand(5)
y1 = torch.rand(4)
z0 = torch.rand(3, 2, 5)
z1 = torch.rand(3, 5, 4)
w = torch.rand(2, 3, 4, 5)
r0 = torch.rand(2, 5)
r1 = torch.rand(3, 5, 4)
r2 = torch.rand(2, 4)
s0 = torch.rand(2, 3, 5, 7)
s1 = torch.rand(11, 3, 17, 5)

# a0 = y0
# a1 = x
# a2 = z0

# b0 = x.T
# b1 = x.T


# identity
a0 = torch.einsum('i', y0)
a1 = torch.einsum('ij', x)
a2 = torch.einsum('ijk', z0)

# permute
b0 = torch.einsum('ij->ji', x)
b1 = torch.einsum('ba', x)

print(z0)
b2 = torch.einsum('jki', z0)
print(b2)
print(z0.shape)
print(b2.shape)
print(z0.permute((1,2,0)).shape)
print(z0.T.shape)
# b2_test = np.transpose(z0, (1,2,0)) #z0[:1,2,0]
# print(b2_test == b2)
1/0

b3 = torch.einsum('ijk - > kij ', z0)
b4 = torch.einsum(' kjil ', w)
b5 = torch.einsum(' ... ij - >... ji ', w)
b6 = torch.einsum(' abc ... - > cba ... ', w)

# trace
c = torch.einsum(' ii ', x)

# sum
d0 = torch.einsum('ij - > ', x)
d1 = torch.einsum('xyz - > ', z0)
d2 = torch.einsum(' ijkl - > ', w)

# sum axis
e0 = torch.einsum('ijk - > i ', z0)
e1 = torch.einsum('ijk - > j ', z0)
e2 = torch.einsum('ijk - > ij ', z0)

# matrix - vector
f0 = torch.einsum('ij ,j - > i ', r0, y0)
f1 = torch.einsum('i , jki - > jk ', y1, r1)

# vector - vector outer product
g0 = torch.einsum('i ,j - > ij ', y0, y1)
g1 = torch.einsum('a ,b ,c ,d - > abcd ', y0, y1, y0, y1)

# batch mm
h0 = torch.einsum('bij , bjk - > bik ', z0, z1)
h1 = torch.einsum('bjk , bij - > bik ', z1, z0)

# bilinear
i = torch.einsum('bn , anm , bm - > ba ', r0, r1, r2)

# tensor contraction
j = torch.einsum(' pqrs , tqvr - > pstv ', s0, s1)
