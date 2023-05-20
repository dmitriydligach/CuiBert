# CuiBert

BERT trained on CUI sequences

## MIMIC preprocessing

* Run Cuis/extract_notes.py to create flat text files with MIMIC notes for processing with cTAKES
* Run cTAKES on these text files to create XMI files
* Run Cuis/process_xmis.py to extract CUIs from XMI files into a single CUI file called cuis.csv
* Run Cuis/long_cuis.py to discard CUIs that are enclosed by other CUIs
* Run MLM/make_samples.py to create sequences for MLM training

## Dr. Bench preprocessing

* Run extract_drbench.py to Dr. Bench CSV files into flat text files for processing with cTAKES
* Run Cuis/process_xmis.py to extract CUIs from XMI files into a single CUI file called cuis.csv
* Run long_cuis.py to discard CUIs that are enclosed by other CUIs
