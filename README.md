# CuiBert

BERT trained on CUI sequences

## MIMIC preprocessing

* Run extract_notes.py to create flat text files with MIMIC notes for processing with cTAKES
* Run cTAKES on these text files to create XMI files
* Run process_xmis.py to extract CUIs from XMI files into a single CUI file called cuis.psv
* Run long_cuis.py to discard CUIs that are enclosed by other CUIs
* Run make_samples.py to create sequences for MLM training

## Dr. Bench preprocessing

* Run extract_drbench.py to Dr. Bench CSV files into flat text files for processing with cTAKES
* Run process_xmis.py to extract CUIs from XMI files into a single CUI file called cuis.psv
* Run long_cuis.py to discard CUIs that are enclosed by other CUIs

## MLM training

* Update max_position_embeddings in config.json
* Update max_seq_length in run.sh
* Possibly update per_device_train/eval_batch_size in run.sh
* Possibly update train_file in run.sh
* Put CuiTokenizer in MLM directory
* Run MLM/run.sh