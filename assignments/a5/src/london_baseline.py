import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse
random.seed(0)

import dataset
import model
import trainer
import utils
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
predictions = []
for line in tqdm(open("birth_dev.tsv")):
    pred="London"
    predictions.append(pred)
total, correct = utils.evaluate_places("birth_dev.tsv", predictions)
if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))