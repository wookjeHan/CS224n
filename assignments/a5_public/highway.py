#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
class Highway(nn.Module):
    def __init__(self, embed_size:int):
        super(Highway,self).__init__()


        self.Wproj=nn.Linear(embed_size,embed_size)
        self.Wgate=nn.Linear(embed_size,embed_size)
    def forward(self,conv_out):
        xProj=torch.relu(self.Wproj(conv_out))
        xGate=torch.sigmoid(self.Wgate(conv_out))
        xHighway=xProj*xGate+(1-xGate)*conv_out
        return xHighway



### END YOUR CODE 

