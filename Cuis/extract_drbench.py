#! /usr/bin/env python3
import pandas, string, os

drbench_path = 'DrBench/Csv/summ_0821_train.csv'
out_dir = 'DrBench/Text/Train/'

def csv_to_flat_files():
  """Convert a csv dr bench file to flat files"""

  data_csv = os.path.join(base_path, drbench_path)
  df = pandas.read_csv(data_csv, dtype='str')

  for file, assm, sum in zip(df['File ID'], df['Assessment'], df['Summary']):
    assess_file = os.path.basename(file) + '-assess.txt'
    sum_file = os.path.basename(file) + '-sum.txt'

    asses_out = open(os.path.join(base_path, out_dir, assess_file), 'w')
    sum_out = open(os.path.join(base_path, out_dir, sum_file), 'w')

    # sometimes assm is empty and pandas returns a float
    if type(assm) == str and type(sum) == str:
      printable_assm = ''.join(c for c in assm if c in string.printable)
      printable_sum = ''.join(c for c in sum if c in string.printable)

      asses_out.write(printable_assm + '\n')
      sum_out.write(printable_sum + '\n')

if __name__ == "__main__":

  base_path = os.environ['DATA_ROOT']
  csv_to_flat_files()
