#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from dataclasses import dataclass

import os, random
import data, utils

# deterministic determinism
torch.manual_seed(2020)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
random.seed(2020)

# model and model config locations
model_path = 'Model/model.pt'
config_path = 'Model/config.p'

@dataclass
class ModelConfig:
  """Everything we need to train"""

  train_data_path: str
  dev_data_path: str
  test_data_path: str
  cui_vocab_size: str

  epochs: int
  batch: int
  hidden: int
  dropout: float
  optimizer: str
  lr: float

class BagOfWords(nn.Module):

  def __init__(
    self,
    input_vocab_size,
    output_vocab_size,
    hidden_units,
    dropout_rate):
    """Construct us"""

    super(BagOfWords, self).__init__()

    self.hidden = nn.Linear(
      in_features=input_vocab_size,
      out_features=hidden_units)

    self.activation = nn.Tanh()

    self.dropout = nn.Dropout(dropout_rate)

    self.classifier = nn.Linear(
      in_features=hidden_units,
      out_features=output_vocab_size)

    self.init_weights()

  def init_weights(self):
    """Never trust pytorch default weight initialization"""

    torch.nn.init.xavier_uniform_(self.hidden.weight)
    torch.nn.init.xavier_uniform_(self.classifier.weight)
    torch.nn.init.zeros_(self.hidden.bias)
    torch.nn.init.zeros_(self.classifier.bias)

  def forward(self, texts):
    """Optionally return hidden layer activations"""

    features = self.hidden(texts)
    output = self.activation(features)
    output = self.dropout(output)
    output = self.classifier(output)

    return output

def make_data_loader(model_inputs, model_outputs, batch_size, partition):
  """DataLoader objects for train or dev/test sets"""

  # e.g. transformers take input ids and attn masks
  if type(model_inputs) is tuple:
    tensor_dataset = TensorDataset(*model_inputs, model_outputs)
  else:
    tensor_dataset = TensorDataset(model_inputs, model_outputs)

  # use sequential sampler for dev and test
  if partition == 'train':
    sampler = RandomSampler(tensor_dataset)
  else:
    sampler = SequentialSampler(tensor_dataset)

  data_loader = DataLoader(
    tensor_dataset,
    sampler=sampler,
    batch_size=batch_size)

  return data_loader

def fit(model, train_loader, val_loader, n_epochs):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  criterion = nn.BCEWithLogitsLoss()

  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.lr)

  best_loss = float('inf')
  optimal_epochs = 0

  for epoch in range(1, n_epochs + 1):

    model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:
      optimizer.zero_grad()

      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_outputs = batch

      logits = model(batch_inputs)
      loss = criterion(logits, batch_outputs)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_tr_loss = train_loss / num_train_steps
    val_loss, val_accuracy = evaluate(model, val_loader)
    print('ep: %d, tr loss: %.4f, val loss: %.4f, val acc: %.4f' % \
          (epoch, av_tr_loss, val_loss, val_accuracy))

    if val_loss < best_loss:
      print('loss improved, saving model...')
      torch.save(model.state_dict(), model_path)
      best_loss = val_loss
      optimal_epochs = epoch

  return best_loss, optimal_epochs

def multi_label_accuracy(pred_labels, true_labels):
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

def evaluate(model, data_loader):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  criterion = nn.BCEWithLogitsLoss()
  total_loss, num_steps = 0, 0

  model.eval()

  all_pred_labels = None
  all_true_labels = None

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_outputs = batch

    with torch.no_grad():
      logits = model(batch_inputs)
      loss = criterion(logits, batch_outputs)

    batch_logits = logits.detach().to('cpu')
    batch_outputs = batch_outputs.to('cpu')
    batch_probs = torch.sigmoid(batch_logits)
    batch_preds = (batch_probs > 0.5).int()

    if all_pred_labels is None:
      all_pred_labels = batch_preds
      all_true_labels = batch_outputs
    else:
      all_pred_labels = torch.cat([all_pred_labels, batch_preds], dim=0)
      all_true_labels = torch.cat([all_true_labels, batch_outputs], dim=0)

    total_loss += loss.item()
    num_steps += 1

  av_loss = total_loss / num_steps
  accuracy = multi_label_accuracy(all_pred_labels, all_true_labels)

  return av_loss, accuracy
 
def model_selection():
  """Eval on the dev set"""

  train_set = data.DatasetProvider(
    cui_file_path=config.train_data_path,
    cui_vocab_size=config.cui_vocab_size,
    tokenize_from_scratch=True)

  dev_set = data.DatasetProvider(
    cui_file_path=config.dev_data_path,
    cui_vocab_size=config.cui_vocab_size,
    tokenize_from_scratch=False)

  tr_in_seqs, tr_out_seqs = train_set.load_as_sequences()
  dev_in_seqs, dev_out_seqs = dev_set.load_as_sequences()
  print('loaded %d training and %d validation samples' % \
        (len(tr_in_seqs), len(dev_in_seqs)))

  max_cui_seq_len = max(len(seq) for seq in tr_in_seqs)
  max_out_seq_len = max(len(seq) for seq in tr_out_seqs)
  print('longest cui input sequence:', max_cui_seq_len)
  print('longest cui ouput sequence:', max_out_seq_len)

  train_loader = make_data_loader(
    utils.sequences_to_matrix(tr_in_seqs, len(train_set.tokenizer.stoi)),
    utils.sequences_to_matrix(tr_out_seqs, len(train_set.tokenizer.stoi)),
    config.batch,
    'train')

  val_loader = make_data_loader(
    utils.sequences_to_matrix(dev_in_seqs, len(train_set.tokenizer.stoi)),
    utils.sequences_to_matrix(dev_out_seqs, len(train_set.tokenizer.stoi)),
    config.batch,
    'dev')

  model = BagOfWords(
    input_vocab_size=len(train_set.tokenizer.stoi),
    output_vocab_size=len(train_set.tokenizer.stoi),
    hidden_units=config.hidden,
    dropout_rate=config.dropout)

  best_loss, optimal_epochs = fit(
    model,
    train_loader,
    val_loader,
    config.epochs)
  print('best loss %.4f after %d epochs' % (best_loss, optimal_epochs))

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  config = ModelConfig(
    train_data_path=os.path.join(base, 'DrBench/Cui/LongestSpan/train.csv'),
    dev_data_path=os.path.join(base, 'DrBench/Cui/LongestSpan/dev.csv'),
    test_data_path=os.path.join(base, 'DrBench/Cui/LongestSpan/test.csv'),
    cui_vocab_size='all',
    epochs=50,
    batch=128,
    hidden=10000,
    dropout=0.25,
    optimizer='Adam',
    lr=1)

  model_selection()
