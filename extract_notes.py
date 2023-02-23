#! /usr/bin/env python3
import pandas, string, os

NOTES_CSV = 'MimicIII/Source/NOTEEVENTS.csv'
OUT_DIR = 'Temp/MimicNotes'
BATCH_SIZE = 500000

def notes_to_flat_files():
  """Split input into files grouped in several directories"""

  notes_csv = os.path.join(base_path, NOTES_CSV)
  out_dir = os.path.join(base_path, OUT_DIR)

  data_frame = pandas.read_csv(notes_csv, dtype='str')

  for id, text in zip(data_frame.ROW_ID, data_frame.TEXT):
    out_path = f'{out_dir}/{int(id) // BATCH_SIZE}/{id}.txt'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    outfile = open(out_path, 'w')
    printable = ''.join(c for c in text if c in string.printable)
    outfile.write(printable)

if __name__ == "__main__":

  base_path = os.environ['DATA_ROOT']
  notes_to_flat_files()
