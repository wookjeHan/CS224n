#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
class Highway(nn.Module):
    def __init__(self,embedding_size):
        super(Highway,self).__init__()

        self.embedding_size=embedding_size
        self.Wproj=nn.Linear(embedding_size,embedding_size)
        self.Wgate=nn.Linear(embedding_size,embedding_size)
    
    def forward(self,xConv):
        xProj=nn.functional.relu(self.Wproj(xConv))#(b,e)
        xGate=torch.sigmoid(self.Wgate(xConv))#(b,e)
        xHighway=xProj*xGate+(1-xGate)*xConv#(b,e)
        return xHighway
### END YOUR CODE 

