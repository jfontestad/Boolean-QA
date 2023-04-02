import torch
import numpy as np

# u = np.full((2, 3), 2)

# print(u)
# print(np.einsum("ij->j", u))
# print(np.einsum("ij->i", u))


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

# b0 = torch.permute(x, (1, 0))
# b1 = torch.permute(x, (1, 0))
# b2 = torch.permute(z0, (2, 0, 1))
# b3 = torch.permute(z0, (2, 0, 1))
# b4 = torch.permute(w, (2, 1, 0, 3))
# b5 = torch.permute(w, (0, 1, 3, 2))
# b6 = torch.permute(w, (2, 1, 0, 3))

# c = torch.trace(x)

# d0 = torch.sum(x)
# d1 = torch.sum(z0)
# d2 = torch.sum(w)

# e0 = torch.sum(z0, dim=(1, 2))
# e1 = torch.sum(z0, dim=(2, 0))
# e2 = torch.sum(z0, dim=2)

# f0 = r0 @ y0
# f1 = torch.sum(y1 * r1, dim=-1)

# g0 = torch.outer(y0, y1)
# g1 = torch.stack([torch.stack([torch.outer(y0, y1) for _ in range(len(y0))], dim=-1) * y0 for _ in range(len(y1))],
#                  dim=-1) * y1

# h0 = torch.bmm(z0, z1)
# h1 = torch.transpose(torch.bmm(torch.transpose(z1, 1, 2), torch.transpose(z0, 1, 2)), 1, 2)

# dot = torch.tensordot(r0, r1, ([1], [1]))
# i = torch.sum(dot * torch.stack([r2 for _ in range(dot.shape[1])], dim=1), dim=-1)

# j = torch.tensordot(s0, s1, ([1, 2], [1, 3]))

# identity
a0 = torch.einsum('i', y0)
a1 = torch.einsum('ij', x)
a2 = torch.einsum('ijk', z0)

# permute
b0 = torch.einsum('ij->ji', x)
b1 = torch.einsum('ba', x)
b2 = torch.einsum('jki', z0)
b3 = torch.einsum('ijk->kij', z0)
b4 = torch.einsum(' kjil ', w)
b5 = torch.einsum('...ij->...ji', w)
b6 = torch.einsum('abc...->cba...', w)

# trace - sum of main diag elements
c = torch.einsum('ii', x)

# sum
d0 = torch.einsum('ij->', x)
d1 = torch.einsum('xyz->', z0)
d2 = torch.einsum('ijkl->', w)

# sum axis
e0 = torch.einsum('ijk->i', z0)
e1 = torch.einsum('ijk->j', z0)
e2 = torch.einsum('ijk->ij', z0)

# matrix - vector
f0 = torch.einsum('ij,j->i', r0, y0)
f1 = torch.einsum('i,jki->jk', y1, r1)

# vector - vector outer product
g0 = torch.einsum('i,j->ij ', y0, y1)
g1 = torch.einsum('a,b,c,d->abcd', y0, y1, y0, y1)

# batch mm
h0 = torch.einsum('bij,bjk->bik', z0, z1)
h1 = torch.einsum('bjk,bij->bik', z1, z0)

# bilinear
i = torch.einsum('bn,anm,bm->ba', r0, r1, r2)

# tensor contraction - a contraction of a and b over multiple dimensions.
j = torch.einsum('pqrs,tqvr->pstv', s0, s1)
