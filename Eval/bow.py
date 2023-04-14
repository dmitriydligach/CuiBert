#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

import os, configparser, random, pickle
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
    dropout_rate,
    save_config=True):
    """Constructor"""

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

    # save configuration for loading later
    if save_config:
      config = {
        'input_vocab_size': input_vocab_size,
        'output_vocab_size': output_vocab_size,
        'hidden_units': hidden_units,
        'dropout_rate': dropout_rate}
      pickle_file = open(config_path, 'wb')
      pickle.dump(config, pickle_file)

  def init_weights(self):
    """Never trust pytorch default weight initialization"""

    torch.nn.init.xavier_uniform_(self.hidden.weight)
    torch.nn.init.xavier_uniform_(self.classifier.weight)
    torch.nn.init.zeros_(self.hidden.bias)
    torch.nn.init.zeros_(self.classifier.bias)

  def forward(self, texts, return_hidden=False):
    """Optionally return hidden layer activations"""

    features = self.hidden(texts)
    output = self.activation(features)
    output = self.dropout(output)
    output = self.classifier(output)

    if return_hidden:
      return features
    else:
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
    print('ep: %d, steps: %d, tr loss: %.4f, val loss: %.4f, val acc: %.4f' % \
          (epoch, num_train_steps, av_tr_loss, val_loss, val_accuracy))

    if val_loss < best_loss:
      print('loss improved, saving model...')
      torch.save(model.state_dict(), model_path)
      best_loss = val_loss
      optimal_epochs = epoch

  return best_loss, optimal_epochs

def multi_label_accuracy(pred_labels, true_labels):
  """Predictions and true labels are multi-hot tensors"""

  correct_predictions = (true_labels * pred_labels).sum()
  total_positive_labels = true_labels.sum()
  accuracy = correct_predictions / total_positive_labels

  return accuracy

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

    if all_pred_labels == None:
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
 
def main():
  """My main main"""

  dp = data.DatasetProvider(config.train_data_path, config.cui_vocab_size)

  in_seqs, out_seqs = dp.load_as_sequences()

  tr_in_seqs, val_in_seqs, tr_out_seqs, val_out_seqs = train_test_split(
    in_seqs, out_seqs, test_size=0.10, random_state=2020)

  print('loaded %d training and %d validation samples' % \
        (len(tr_in_seqs), len(val_in_seqs)))

  max_cui_seq_len = max(len(seq) for seq in tr_in_seqs)
  print('longest cui sequence:', max_cui_seq_len)

  max_code_seq_len = max(len(seq) for seq in tr_out_seqs)
  print('longest code sequence:', max_code_seq_len)

  train_loader = make_data_loader(
    utils.sequences_to_matrix(tr_in_seqs, len(dp.tokenizer.stoi)),
    utils.sequences_to_matrix(tr_out_seqs, len(dp.tokenizer.stoi)),
    config.batch,
    'train')

  val_loader = make_data_loader(
    utils.sequences_to_matrix(val_in_seqs, len(dp.tokenizer.stoi)),
    utils.sequences_to_matrix(val_out_seqs, len(dp.tokenizer.stoi)),
    config.batch,
    'dev')

  model = BagOfWords(
    input_vocab_size=len(dp.tokenizer.stoi),
    output_vocab_size=len(dp.tokenizer.stoi),
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
    train_data_path=os.path.join(base, 'DrBench/Cui/train.csv'),
    test_data_path=os.path.join(base, 'DrBench/Cui/dev.csv'),
    cui_vocab_size='all',
    epochs=100,
    batch=32,
    hidden=512,
    dropout=0.1,
    optimizer='Adam',
    lr=1e-3)

  main()
