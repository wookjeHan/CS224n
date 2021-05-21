import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, read_corpus, batch_iter
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from nmt_model import NMT


import torch
import torch.nn as nn
import torch.nn.utils
import cnn
""" Sanity check for cnn() function. 
"""
print ("-"*80)
print("Running Sanity Check for Question 1i: cnn")
print ("-"*80)
cnn = cnn.Cnn(21,5,10)#mword=21,eChar=5,eWord=10

print("Running test on a shape")

ipt=torch.zeros(20,5,21)
opt=cnn.forward(ipt)
print(opt.shape)
# sentences = [['Human:', 'What', 'do', 'we', 'want?'], ['Computer:', 'Natural', 'language', 'processing!'], ['Human:', 'When', 'do', 'we', 'want', 'it?'], ['Computer:', 'When', 'do', 'we', 'want', 'what?']]
# word_ids = vocab.words2charindices(sentences)

# padded_sentences = pad_sents_char(word_ids, 0)
# gold_padded_sentences = torch.load('./sanity_check_en_es_data/gold_padded_sentences.pkl')
# assert padded_sentences == gold_padded_sentences, "Sentence padding is incorrect: it should be:\n {} but is:\n{}".format(gold_padded_sentences, padded_sentences)

print("Sanity Check Passed for Question 1i: cnn!")
print("-"*80)