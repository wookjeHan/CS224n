#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        
        self.pad_token_idx=vocab.char2id['<pad>']
        self.vocab=vocab
        self.eWord=embed_size
        self.eChar=50
        self.embeddings=nn.Embedding(len(vocab.char2id),self.eChar,padding_idx=self.pad_token_idx)
        self.dropout=nn.Dropout(0.3)
        self.mWord=21
        self.Cnn=CNN(self.eWord,self.mWord,self.eChar)
        self.HighWay=Highway(self.eWord)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        
        embeddedInput=self.embeddings(input)#(s_l,b,mWord,eChar)
        s_l,b,mWord,eChar=embeddedInput.shape
        Xreshaped=embeddedInput.transpose(2,3)#(s_l,b,eChar,mWord)
        Xreshaped=Xreshaped.view(s_l*b,eChar,-1)
        Xconv=self.Cnn(Xreshaped)#(s_l*b,eWord)
        
        # print("===================Xconv shape==============")
        # print(Xconv.shape)
        
        XHighway=self.HighWay(Xconv)#(s_l*b,eWord)
        
        # print("===================XHighway shape============")
        # print(XHighway.shape)
        
        XWordEmb=self.dropout(XHighway)
        XWordEmb=XWordEmb.view(s_l,b,self.eWord)


        ### YOUR CODE HERE for part 1j
        
        return XWordEmb

        ### END YOUR CODE

