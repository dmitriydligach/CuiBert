#!/usr/bin/env python3

import configparser, os, pandas, sys, pathlib, shutil, pickle
import tokenizer
from collections import defaultdict

model_dir = 'Model/'
alphabet_pickle = 'Model/alphabet.p'

class DatasetProvider:
  """Summarization and assessment data"""

  def __init__(self,
               cui_file_path,
               cui_vocab_size):
    """Construct it"""

    self.cui_file_path = cui_file_path

    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # key: file name, value: list of cuis
    self.inputs = defaultdict(list)
    self.outputs = defaultdict(list)

    # pair inputs and outputs
    self.read_data()

    # index cuis
    self.tokenizer = tokenizer.Tokenizer(
      n_words=None if cui_vocab_size == 'all' else int(cui_vocab_size),
      lower=False)
    self.tokenize()

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

    self.tokenizer.fit_on_texts(
      list(self.inputs.values()) +
      list(self.outputs.values()))

    pickle_file = open('Model/tokenizer.p', 'wb')
    pickle.dump(self.tokenizer, pickle_file)
    print('input vocab:', len(self.tokenizer.stoi))

  def load_as_sequences(self):
    """Make x and y"""

    x = []
    y = []

    for file in self.inputs.keys():
      if len(self.inputs[file]) > 0 and len(self.outputs[file]) > 0:
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

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dp = DatasetProvider(
    os.path.join(base, cfg.get('data', 'train')),
    cfg.get('args', 'cui_vocab_size'))

  inputs, outputs = dp.load_as_sequences()
  print(inputs[2])
  print(outputs[2])
