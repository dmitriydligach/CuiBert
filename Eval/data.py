#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')
import tokenizer

import os, shutil, pickle
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# maximum sequence length
max_length = 100

# output CUI tokenizer
default_tokenizer_pickle = 'OutputTokenizer/tokenizer.p'

class SummarizationDataset(Dataset):
  """Read data from files and make inputs/outputs"""

  def __init__(
   self,
   cui_file_path,
   pretrained_tokenizer_path,
   tokenize_from_scratch):
    """Load tokenizer and save corpus path"""

    self.x = []
    self.y = []

    # pair inputs and outputs
    self.read_data(cui_file_path)

    # use tokenizer from the model
    self.pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_path)

    # tokenizer to index possible CUI outputs
    if tokenize_from_scratch:
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
        print(f'skpping {file}: empty assessment or treatment...')
        continue
      self.x.append(inputs[file])
      self.y.append(outputs[file])

  def index_output_cuis(self):
    """Read text and map tokens to ints"""

    if os.path.isdir('OutputTokenizer/'):
      shutil.rmtree('OutputTokenizer/')
    os.mkdir('OutputTokenizer/')

    # index CUIs using outputs
    self.output_tokenizer.fit_on_texts(self.y)

    pickle_file = open(default_tokenizer_pickle, 'wb')
    pickle.dump(self.output_tokenizer, pickle_file)
    print('output vocab size:', len(self.output_tokenizer.stoi))

  def __len__(self):
    """Requried by pytorch"""

    assert(len(self.x) == len(self.y))
    return len(self.x)

  def __getitem__(self, index):
    """Required by pytorch"""

    # sequence of CUI indices + special tokens
    input = self.pretrained_tokenizer(
      self.x[index],
      is_split_into_words=True,
      max_length=max_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    # sequence of CUI indices
    # output = self.pretrained_tokenizer(
    #   self.y[index],
    #   is_split_into_words=True,
    #   add_special_tokens=False)
    #
    # sequences of CUI indices
    output = self.output_tokenizer.texts_to_seqs(
      [self.y[index]],
      add_cls_token=False,
      use_unk_token=False)

    # mult-hot vectors
    labels = torch.zeros(len(self.output_tokenizer.stoi))
    labels[output[0]] = 1.0

    return dict(
      input_ids = input.input_ids.squeeze(),
      attention_mask=input.attention_mask.squeeze(),
      labels = labels)

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  data_path = os.path.join(base, 'DrBench/Cui/LongestSpan/train.csv')

  data = SummarizationDataset(data_path, 'checkpoint-200000')
  print(data[0])