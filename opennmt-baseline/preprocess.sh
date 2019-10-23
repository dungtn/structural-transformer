#!/bin/bash

data_dir=/mnt/nfs/scratch1/dthai/amred-qa/data/$1/data
mkdir -p ${data_dir}/baseline-data
data_prefix=${data_dir}/baseline-data/gq

python preprocess.py -train_src ${data_dir}/training_source_bpe \
                       -train_tgt ${data_dir}/training_target_bpe \
                       -valid_src ${data_dir}/dev_source_bpe  \
                       -valid_tgt ${data_dir}/dev_target_bpe \
                       -save_data ${data_prefix} \
                       -src_vocab_size 30000  \
                       -tgt_vocab_size 30000 \
                       -src_seq_length 10000 \
                       -tgt_seq_length 10000 \
                       -share_vocab
