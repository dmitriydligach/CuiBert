#! /usr/bin/env python3
import os, numpy

# Create samples for pretraining transformers using MIMIC notes
# For now each note is one sample, but will break down further later

base = os.environ['DATA_ROOT']

cui_file_path = '../Cuis/filtered.csv'
out_file_path = './training_data.txt'

min_cui_count = 5
max_cui_count = 25

class UmlsConcept:
  """UMLS concept"""

  def __init__(self, s):
    """E.g. 109143.txt|C0030471|Sinus|snomedct_us|Nasal sinus|0|5"""

    elements = s.split('|')

    if len(elements) != 7:
      raise ValueError('not enough elements:', s)
    else:
      self.file = elements[0]
      self.cui = elements[1]
      self.text  = elements[2]
      self.vocabulary = elements[3]
      self.preferred = elements[4]
      self.start = int(elements[5])
      self.end = int(elements[6])

def main():
  """Main street"""

  out_file = open(out_file_path, 'w')

  cuis = []
  cui_counts = []
  cur_file = None

  for line in open(cui_file_path):
    concept = UmlsConcept(line.strip())

    # are we processing a new file?
    if cur_file != concept.file:
      cui_counts.append(len(cuis))

      if min_cui_count <= len(cuis) <= max_cui_count:
        out_file.write(' '.join(cuis) + '\n')

      cur_file = concept.file
      cuis = []

    cuis.append(concept.cui[1:])

  print(f'average number of cuis: {numpy.mean(cui_counts):.2f}')
  print(f'median number of cuis: {numpy.median(cui_counts):.2f}')
  print(f'standard deviation: {numpy.std(cui_counts):.2f}')
  print(f'max number of cuis: {numpy.max(cui_counts)}')

if __name__ == "__main__":

  main()
