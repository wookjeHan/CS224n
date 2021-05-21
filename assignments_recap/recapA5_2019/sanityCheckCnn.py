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
cnn = cnn.CNN(21,5,10)#eWord=21,mWord=5,eChar=10

print("Running test on a shape")

ipt=torch.zeros(20,10,5)
opt=cnn.forward(ipt)
print(opt.shape)

print("Sanity Check Passed for Question 1i: cnn!")
print("-"*80)