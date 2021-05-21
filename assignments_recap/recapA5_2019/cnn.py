#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self,eWord,mWord,eChar,k=5):
        #eWord is word embedding size
        #mWord is char per word
        #eChar is character embedding size
        super(CNN,self).__init__()
        
        self.k=k
        self.eWord=eWord
        self.mWord=mWord
        self.eChar=eChar

        self.conv=nn.Conv1d(in_channels=eChar,out_channels=eWord,kernel_size=k)
    
    def forward(self,xRes):
        xConv=nn.functional.relu(self.conv(xRes))
        xConv,_=torch.max(xConv,dim=-1)
        return xConv
### END YOUR CODE 

