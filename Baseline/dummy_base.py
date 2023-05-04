#!/usr/bin/env python3

import os
import data

base = os.environ['DATA_ROOT']
train_data_path = os.path.join(base, 'DrBench/Cui/LongestSpan/train.csv')
dev_data_path = os.path.join(base, 'DrBench/Cui/LongestSpan/dev.csv')

# Example:
# ['C1956346', 'C0018946', 'C0038615'] ['C0038615', 'C0032285']
# Intersection:          1 CUI
# Precision denominator: 3 (input size)
# Recall denominator:    2 (output size)

def main():
  """Just predict all CUIs in the input"""

  dp = data.DatasetProvider(
    cui_file_path=dev_data_path,
    cui_vocab_size='all',
    tokenize_from_scratch=True)

  total_correct = 0
  total_prediction = 0
  total_gold = 0

  for file_name in dp.inputs.keys():
    prediction = set(dp.inputs[file_name])
    gold = set(dp.outputs[file_name])
    intersection = prediction.intersection(gold)

    total_correct += len(intersection)
    total_prediction += len(prediction)
    total_gold += len(gold)

  precision = total_correct / total_prediction
  recall = total_correct / total_gold
  f1 = 2 * (precision * recall) / (precision + recall)

  print('precision:', precision)
  print('recall:', recall)
  print('f1 score:', f1)

if __name__ == "__main__":

  main()