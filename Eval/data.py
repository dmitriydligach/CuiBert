#!/usr/bin/env python3

import os, pathlib, numpy
from collections import defaultdict

from torch.utils.data import Dataset
from transformers import AutoTokenizer

# maximum sequence length
max_length = 25

class SummarizationDataset(Dataset):
  """Read data from files and make inputs/outputs"""

  def __init__(self, cui_file_path, tokenizer_path):
    """Load tokenizer and save corpus path"""

    self.x = []
    self.y = []

    self.cui_file_path = cui_file_path

    # pair inputs and outputs
    self.read_data()

    # use tokenizer from the model
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

  def __len__(self):
    """Requried by pytorch"""

    assert(len(self.x) == len(self.y))
    return len(self.x)

  def __getitem__(self, index):
    """Required by pytorch"""

    input = self.tokenizer(
      self.x[index],
      is_split_into_words=True,
      max_length=max_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    output = self.tokenizer(
      self.y[index],
      is_split_into_words=True,
      max_length=max_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    return dict(
      input_ids = input.input_ids.squeeze(),
      attention_mask=input.attention_mask.squeeze(),
      labels = output.input_ids.squeeze())

  def read_data(self):
    """Read a CUI file and pair assessments and plans"""

    # key: file name, value: list of cuis
    inputs = defaultdict(list)
    outputs = defaultdict(list)

    # read CUI file
    for line in open(self.cui_file_path):
      file, cui, _, _, _, _, _ = line.split('|')
      mimic_file, assess_or_sum = file.split('-')

      if assess_or_sum == 'assess.txt':
        inputs[mimic_file].append(cui[1:])
      else:
        outputs[mimic_file].append(cui[1:])

    # pair inputs and outputs
    for file in inputs.keys():
      if len(inputs[file]) > 0 and len(outputs[file]) > 0:
        self.x.append(inputs[file])
        self.y.append(outputs[file])
      else:
        print('empty assessment or treament:', file)

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  data_path = os.path.join(base, 'DrBench/Cui/LongestSpan/train.csv')

  data = SummarizationDataset(data_path, 'checkpoint-30000')
  print(data[0])