#!/usr/bin/env bash

# from readme:
# If your dataset is organized with one sample per line, you can
# use the `--line_by_line` flag (otherwise the script concatenates all
# texts and then splits them in blocks of the same length).

python3 run_mlm.py \
    --line_by_line \
    --model_type bert \
    --config_name config.json \
    --tokenizer_name CuiTokenizer \
    --train_file ../Data/mlm-training_data.txt \
    --max_seq_length 100 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --do_train \
    --do_eval \
    --output_dir Output \
    --overwrite_output_dir \
    --learning_rate 5e-05 \
    --num_train_epochs 500 \
    --max_steps -1 \
    --log_level passive \
    --logging_dir TensorboardLogs \
    --save_steps 10000 \
    --disable_tqdm True \
    --logging_strategy epoch \
    --evaluation_strategy epoch
