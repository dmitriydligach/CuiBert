#!/usr/bin/env python3

import collections

class Tokenizer:
  """Tokenization and vectorization"""

  def __init__(self, n_words):
    """Construction deconstruction"""

    self.stoi = {}
    self.itos = {}
    self.counts = collections.Counter()
    self.n_words = n_words

  def fit_on_texts(self, texts):
    """Fit on documents represented as pre-tokenized lists of tokens"""

    for token_list in texts:
      self.counts.update(token_list)

    index = 0
    for token, _ in self.counts.most_common(self.n_words):
      self.stoi[token] = index
      self.itos[index] = token
      index += 1

  def texts_to_seqs(self, texts):
    """Lists of tokens to lists of int sequences"""

    int_sequences = []
    for token_list in texts:

      int_sequence = []
      for token in token_list:
        if token in self.stoi:
          int_sequence.append(self.stoi[token])

      int_sequences.append(int_sequence)

    return int_sequences

if __name__ == "__main__":

  sents = ['it is happening again',
           'the owls are not what they seem',
           'again and again',
           'the owls are happening']

  tokenizer = Tokenizer(n_words=6)

  sents = [sent.split() for sent in sents]

  tokenizer.fit_on_texts(sents)
  print('counts:', tokenizer.counts)
  print('stoi:', tokenizer.stoi)
  print('itos:', tokenizer.itos)

  seqs = tokenizer.texts_to_seqs(sents)
  print(seqs)