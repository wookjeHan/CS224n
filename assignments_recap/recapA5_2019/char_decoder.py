#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder,self).__init__()
        self.target_vocab=target_vocab
        self.charDecoder=nn.LSTM(char_embedding_size,hidden_size)
        self.char_output_projection=nn.Linear(hidden_size,len(target_vocab.char2id))
        self.decoderCharEmb=nn.Embedding(len(target_vocab.char2id),char_embedding_size,padding_idx=target_vocab.char2id['<pad>'])

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        # print('=========input==============')
        # print(input.shape)

        inputLength,batch=input.shape
        input=self.decoderCharEmb(input)#(length,batch,embedded)
        output,dec_hidden=self.charDecoder(input,dec_hidden)#ouput shape->(length,batch,hidden) dec_hidden is tuple of (1,batch,hidden)

        # print('===============output==============')
        # print(output.shape)
        # print('===============dec_hidden==========')
        # a,b=dec_hidden
        # print(a.shape,b.shape)
        # print('=============score==============')

        scores=self.char_output_projection(output)#scores shape ->(length,batch,Vchar)
        # print(scores.shape)

        return scores,dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        trainInput=char_sequence[:-1]#(length-1,batch)
        goldTarget=char_sequence[1:]#(length-1,batch)
        
        length_1,batch=trainInput.shape

        scores,dec_hidden=self.forward(trainInput,dec_hidden)#scores shape->(length-1,batch,Vchar)
        loss=nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'],reduction='sum')

        scores=scores.view(length_1*batch,-1)
        goldTarget=goldTarget.reshape(length_1*batch)
        output=loss(scores,goldTarget)
        return output
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        
        _,batch,hidden=initialStates[0].shape
        dec_hidden=initialStates#(1,batch,hidden)
        decodedWords=[]
        
        answers=torch.tensor([self.target_vocab.start_of_word]*batch,device=device) #(batch,)
        answers=answers.view(-1,1)#(batch,1)
        inputChars=answers.view(1,-1)#(1,batch)
        for i in range(max_length):
            scores,dec_hidden=self.forward(inputChars,dec_hidden)#scores is (1,batch,vChar)
            scores=scores.view(scores.shape[1],-1)#(batch,vChar)
            nextChars=torch.argmax(scores,1)#(batch,)
            nextChars=nextChars.view(-1,1)#(batch,1)
            inputChars=nextChars.view(1,-1)#(1,batch)
            
            # print('==================nextChars shape==========')
            # print(nextChars.shape)
            answers=torch.cat((answers,nextChars),1)#(batch,2)....
            # print('=================answers shape=============')
            # print(answers.shape)
        
        for i in range(batch):
            word=''
            for j in range(max_length):
                newChar=answers[i][j+1]
                if newChar==self.target_vocab.end_of_word:
                    # print("Breakkkkkkkkkkkkkkkkkkkk")
                    break
                word=word+self.target_vocab.id2char[answers[i,j+1].item()]
                # print("=============Not Yet================")
                # print(word)
            # print("=============word adding============")
            # print(word)
            decodedWords.append(word)
        # print("==============decodedWords==========")
        # print(decodedWords)
        return decodedWords
        


        ### END YOUR CODE

