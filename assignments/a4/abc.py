#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.utils

seq=torch.tensor([[1,2,0], [3,0,0], [4,5,6]])

a=seq.unsqueeze(2)
b=seq.unsqueeze(-1)

print(a)
print(b)
print(a==b)
