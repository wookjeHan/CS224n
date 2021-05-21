#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
### YOUR CODE HERE for part 1i
class Cnn(nn.Module):
    def __init__(self,mWord:int,eChar:int,eWord:int):
        super(Cnn,self).__init__()

        self.convLayer=nn.Conv1d(in_channels=eChar,out_channels=eWord,kernel_size=5)
        self.maxPool=nn.MaxPool1d(kernel_size=mWord-4)
    def forward(self,x):
        xConv=self.convLayer(x)
        xConv_out=self.maxPool(torch.relu(xConv))
        xConv_out=torch.squeeze(xConv_out,dim=2)
        return xConv_out
### END YOUR CODE

