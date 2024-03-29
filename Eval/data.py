#!/usr/bin/env python3

import sys

import utils

sys.path.append('../Lib/')
import tokenizer

import os, shutil, pickle
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# output CUI tokenizer
default_tokenizer_pickle = 'OutputTokenizer/tokenizer.p'

class SummarizationDataset(Dataset):
  """Read data from files and make inputs/outputs"""

  def __init__(
   self,
   cui_file_path,
   max_input_length,
   pretrained_tokenizer,
   tokenize_output_from_scratch):
    """Load tokenizer and save corpus path"""

    self.x = []
    self.y = []

    # set it to whatever was used for pretraining
    self.max_input_length = max_input_length

    # pair inputs and outputs
    self.read_data(cui_file_path)

    # for inputs, use tokenizer from pretraining
    self.input_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    # tokenizer to index possible CUI outputs
    if tokenize_output_from_scratch:
      self.output_tokenizer = tokenizer.Tokenizer(n_words=None)
      self.index_output_cuis()
    else:
      pkl = open(default_tokenizer_pickle, 'rb')
      self.output_tokenizer = pickle.load(pkl)

  def read_data(self, cui_file_path):
    """Read a CUI file and pair assessments and plans"""

    # key: file name, value: list of cuis
    inputs = defaultdict(list)
    outputs = defaultdict(list)

    # read CUI file
    for line in open(cui_file_path):
      file, cui, _, _, _, _, _ = line.split('|')
      mimic_file, assess_or_sum = file.split('-')

      if assess_or_sum == 'assess.txt':
        inputs[mimic_file].append(cui[1:])
      else:
        outputs[mimic_file].append(cui[1:])

    # pair inputs and outputs
    for file in inputs.keys():
      if len(inputs[file]) == 0 or len(outputs[file]) == 0:
        # print(f'skpping {file}: empty assessment or treatment...')
        continue
      self.x.append(inputs[file])
      self.y.append(outputs[file])

  def index_output_cuis(self):
    """Map output cuis to ints"""

    if os.path.isdir('OutputTokenizer/'):
      shutil.rmtree('OutputTokenizer/')
    os.mkdir('OutputTokenizer/')

    # index CUIs using outputs
    self.output_tokenizer.fit_on_texts(self.y)

    pickle_file = open(default_tokenizer_pickle, 'wb')
    pickle.dump(self.output_tokenizer, pickle_file)
    print('output vocab size:', len(self.output_tokenizer.stoi))

  def compute_weights(self):
    """Weights for training"""

    out_seqs = self.output_tokenizer.texts_to_seqs(self.y)
    out_multi_hot = utils.sequences_to_matrix(
      out_seqs,
      len(self.output_tokenizer.stoi))
    weights = 1 / (torch.sum(out_multi_hot, dim=0) + 1)

    return weights

  def __len__(self):
    """Requried by pytorch"""

    assert(len(self.x) == len(self.y))
    return len(self.x)

  def __getitem__(self, index):
    """Required by pytorch"""

    # input as a sequence of CUI indices + special tokens
    input = self.input_tokenizer(
      self.x[index],
      is_split_into_words=True,
      add_special_tokens=True,
      max_length=self.max_input_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    # output as a sequences of CUI indices
    output = self.output_tokenizer.texts_to_seqs([self.y[index]])[0]
    # output as a multi-hot vector
    labels = torch.zeros(len(self.output_tokenizer.stoi))
    labels[output] = 1.0

    return dict(
      input_ids = input.input_ids.squeeze(),
      attention_mask=input.attention_mask.squeeze(),
      labels = labels)

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  data_path = os.path.join(base, 'DrBench/Cui/LongestSpan/train.csv')

  model_path = '/home/dima/Git0/CuiBert/MLM/Output/checkpoint-60000/'
  data = SummarizationDataset(data_path, 100, model_path, True)
  print(data[11])

  data.compute_weights()