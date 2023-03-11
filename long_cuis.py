#!/usr/bin/env python3

input_file = 'cuis.csv'
output_file = 'filtered.csv'

class UmlsConcept:
  """UMLS concept"""

  def __init__(self, s):
    """From a string"""

    elements = s.split('|')

    self.file = elements[0]
    self.cui = elements[1]
    self.text  = elements[2]
    self.vocabulary = elements[3]
    self.preferred = elements[4]
    self.start = int(elements[5])
    self.end = int(elements[6])

  def __str__(self):
    """Convert back to string"""

    as_list = (
      self.file,
      self.cui,
      self.text,
      self.vocabulary,
      self.preferred,
      str(self.start),
      str(self.end))

    return '|'.join(as_list)

  def is_contained(self, another):
    """Return true if self should be eliminated"""

    # comparison to itself
    if self == another:
      return False

    # same span different CUIs
    if self.start == another.start and \
       self.end == another.end and \
       self.cui != another.cui:
      return False

    if self.start >= another.start and \
       self.end <= another.end:
      return True

    return False

def save_longest_spans(concepts, out_file):
  """Preserve only longest concept mentions"""

  # compare concept a to all other concepts
  # discared if it lies inside another concept
  discard_flag = False
  for a in concepts:
    for b in concepts:
      if a.is_contained(b):
        discard_flag = True
        break

    if not discard_flag:
      out_file.write(str(a) + '\n')

def main():
  """Main street"""

  cur_file = None
  concepts = []

  out_file = open(output_file, 'w')

  for line in open(input_file):
    concept = UmlsConcept(line.strip())
    if cur_file != concept.file:
      cur_file = concept.file
      save_longest_spans(concepts, out_file)
      concepts = []

    concepts.append(concept)

if __name__ == "__main__":

    main()