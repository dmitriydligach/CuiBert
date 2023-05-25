#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import os, shutil, pickle
from collections import defaultdict
from dataclasses import dataclass
import tokenizer

default_tokenizer_pickle = 'Model/tokenizer.p'

@dataclass
class ModelConfig:
  """Everything we need to train"""

  train_data_path: str
  test_data_path: str
  cui_vocab_size: str

class DatasetProvider:
  """Summarization and assessment data"""

  def __init__(self,
               cui_file_path,
               cui_vocab_size,
               tokenize_from_scratch):
    """Construct it"""

    self.cui_file_path = cui_file_path

    # key: file name, value: list of cuis
    self.inputs = defaultdict(list)
    self.outputs = defaultdict(list)

    # pair inputs and outputs
    self.read_data()

    # do we need to index CUIs?
    if tokenize_from_scratch:
      self.tokenizer = tokenizer.Tokenizer(
        n_words=None if cui_vocab_size == 'all' else cui_vocab_size)
      self.tokenize()
    else:
      pkl = open(default_tokenizer_pickle, 'rb')
      self.tokenizer = pickle.load(pkl)

  def read_data(self):
    """Read a CUI file"""

    for line in open(self.cui_file_path):
      file, cui, _, _, _, _, _ = line.split('|')
      mimic_file, assess_or_sum = file.split('-')

      if assess_or_sum == 'assess.txt':
        self.inputs[mimic_file].append(cui)
      else:
        self.outputs[mimic_file].append(cui)

  def tokenize(self):
    """Read text and map tokens to ints"""

    if os.path.isdir('Model/'):
      shutil.rmtree('Model/')
    os.mkdir('Model/')

    # index CUIs using inputs and outputs
    self.tokenizer.fit_on_texts(
      list(self.inputs.values()) +
      list(self.outputs.values()))

    pickle_file = open(default_tokenizer_pickle, 'wb')
    pickle.dump(self.tokenizer, pickle_file)
    print('input vocab:', len(self.tokenizer.stoi))

  def load_as_sequences(self):
    """Make x and y"""

    x = []
    y = []

    for file in self.inputs.keys():
      # if len(self.inputs[file]) > 0 and len(self.outputs[file]) > 0:
      x.append(self.inputs[file])
      y.append(self.outputs[file])

    # make x and y matrices
    x = self.tokenizer.texts_to_seqs(x, add_cls_token=False)
    y = self.tokenizer.texts_to_seqs(y, add_cls_token=False)

    # column zero is empty
    # return x, y[:,1:]
    return x, y

if __name__ == "__main__":
  """Test dataset class"""

  base = os.environ['DATA_ROOT']
  config = ModelConfig(
    train_data_path=os.path.join(base, 'DrBench/Cui/LongestSpan/train.csv'),
    test_data_path=os.path.join(base, 'DrBench/Cui/LongestSpan/dev.csv'),
    cui_vocab_size='all')

  dp = DatasetProvider(
    cui_file_path=config.train_data_path,
    cui_vocab_size=config.cui_vocab_size,
    tokenize_from_scratch=True)

  inputs, outputs = dp.load_as_sequences()
  print(inputs[2])
  print(outputs[2])
