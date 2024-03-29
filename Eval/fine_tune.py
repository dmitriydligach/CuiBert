#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch, random, data, os, shutil, metrics
from transformers import (TrainingArguments,
                          Trainer,
                          AutoModelForSequenceClassification,
                          IntervalStrategy)

# misc constants
pretrained_model_path = '/home/dima/Git0/CuiBert/MLM/Output/checkpoint-90000'
output_dir = './Results'
metric_for_best_model = 'eval_multilab_f1'
tokenizer_path = pretrained_model_path
results_file = './results.txt'
max_input_length = 100 # must match what was used for pretraining

# hyperparameters
model_selection_n_epochs = 100
batch_size = 512

# search over these hyperparameters
classifier_dropouts = [0.1, 0.25, 0.5]
learning_rates = [1, 1e-1, 1e-2, 5e-2, 1e-3, 1e-4, 1e-5]

class WeightedTrainer(Trainer):
  """ Need this to compute weighted loss"""

  def __init__(
    self,
    model,
    args,
    train_dataset,
    eval_dataset,
    compute_metrics,
    weights=None):
    """Deconstruct the constructor"""

    super(WeightedTrainer, self).__init__(
      model=model,
      args=args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      compute_metrics=compute_metrics)

    self.weights = weights

  def compute_loss(self, model, inputs, return_outputs=False):
    """Weighted loss"""

    loss_fct = torch.nn.BCEWithLogitsLoss(reduction='none')

    model_outputs = model(**inputs)
    labels = inputs.get('labels')
    logits = model_outputs.get('logits')
    loss = loss_fct(logits, labels)

    if self.weights is None:
      loss = loss.mean()
    else:
      weights = self.weights.to(model.device)
      loss = (loss * weights).mean()

    return (loss, model_outputs) if return_outputs else loss

def compute_metrics(eval_pred):
  """Compute custom evaluation metric"""

  logits, labels = eval_pred

  logits = torch.from_numpy(logits)
  labels = torch.from_numpy(labels)

  probs = torch.sigmoid(logits)
  preds = (probs > 0.5).int()

  # https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
  return {'multilab_f1': metrics.f1(preds, labels)}

def grid_search(train_path, dev_path):
  """Try different hyperparameter combinations and return the best"""

  # key: performance, value: [best num of epochs, learning rate]
  search_results = {}

  for classifier_dropout in classifier_dropouts:
    for learning_rate in learning_rates:

      print('evaluating lr and dropout:', learning_rate, classifier_dropout)
      best_n_epochs, best_metric_value = eval_on_dev_set(
        train_path,
        dev_path,
        learning_rate,
        classifier_dropout)

      search_results[best_metric_value] = \
        [best_n_epochs, learning_rate, classifier_dropout]

  print('search results:', search_results)
  best_performance = max(search_results.keys())
  print('best performance:', best_performance)
  optimal_n_epochs, optimal_learning_rate, optimal_classifier_dropout = \
    search_results[best_performance]
  print('optimal epochs, lr, dropout:',
    optimal_n_epochs, optimal_learning_rate, optimal_classifier_dropout)

  return optimal_n_epochs, optimal_learning_rate, optimal_classifier_dropout

def eval_on_dev_set(train_path, dev_path, learning_rate, classifier_dropout):
  """Make a dev set, fine-tune, and evaluate on it"""

  # deterministic determinism
  torch.manual_seed(2022)
  random.seed(2022)

  train_dataset = data.SummarizationDataset(
    train_path,
    max_input_length,
    tokenizer_path,
    tokenize_output_from_scratch=True)
  dev_dataset = data.SummarizationDataset(
    dev_path,
    max_input_length,
    tokenizer_path,
    tokenize_output_from_scratch=False)

  model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_path,
    num_labels=len(train_dataset.output_tokenizer.stoi),
    problem_type='multi_label_classification')
  model.dropout = torch.nn.modules.dropout.Dropout(classifier_dropout)

  training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=model_selection_n_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    load_best_model_at_end=True,
    metric_for_best_model=metric_for_best_model,
    save_strategy=IntervalStrategy.EPOCH,
    evaluation_strategy=IntervalStrategy.EPOCH,
    disable_tqdm=True)
  trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    weights=train_dataset.compute_weights())
  trainer.train()

  best_n_epochs = None
  best_metric_value = trainer.state.best_metric

  for entry in trainer.state.log_history:
    if metric_for_best_model in entry:
      print(f"ep: {entry['epoch']}, perf: {entry[metric_for_best_model]}")
      if entry[metric_for_best_model] == best_metric_value:
        best_n_epochs = entry['epoch']
  print(f"best epochs: {best_n_epochs}, best performance: {best_metric_value}")

  # remove intermediate checkpoint dir to save space
  shutil.rmtree(output_dir)

  return best_n_epochs, best_metric_value

def main():
  """Evaluate on a few datasets"""

  base_path = os.environ['DATA_ROOT']
  train_path = os.path.join(base_path, 'DrBench/Cui/LongestSpan/train.csv')
  dev_path = os.path.join(base_path, 'DrBench/Cui/LongestSpan/dev.csv')

  grid_search(train_path, dev_path)

if __name__ == "__main__":
  "My kind of street"

  main()
