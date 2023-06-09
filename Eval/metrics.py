#!/usr/bin/env python3

def f1(pred_labels, true_labels):
  """Predictions and true labels are multi-hot tensors"""

  # true_labels = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0]]
  # pred_labes =  [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]]
  # recall = 4 / 7 = 0.57
  # precision = 4 / 6 = 0.67

  correct_predictions = (true_labels * pred_labels).sum()
  total_positive_labels = true_labels.sum()
  total_predicted_labels = pred_labels.sum()

  if total_predicted_labels == 0 or total_positive_labels == 0:
    return 0

  precision = correct_predictions / total_predicted_labels
  recall = correct_predictions / total_positive_labels
  f1 = 2 * (precision * recall) / (precision + recall)

  return f1

if __name__ == "__main__":

  print()
