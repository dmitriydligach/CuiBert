# Training CUI BERT

- run train_tokenizer.py which traines a tokenizer on text
- run cui_tokenizer.py which replaces the vocabulary with CUIs ('C' stripped)
- copy cui_tokenizer.json to Tokenizer/tokenizer.json
- remove Tokenizer/vocab.txt just in case
- ensure tokenizer dir is called CuiTokenizer
- replace tokenizer_name in run.sh with CuiTokenizer
