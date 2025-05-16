import torch
import numpy as np
t1 = torch.rand(2,3,4)
n = np.ones([2,3,5],dtype = np.float32)
print(t1)

t2 = torch.from_numpy(n)
print(t2)

res = t1.matmul(t2.T)
print(res.dtype,res.shape)