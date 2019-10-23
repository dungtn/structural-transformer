#!/bin/bash

data_dir=/mnt/nfs/scratch1/dthai/amred-qa/data/$1/data
data_prefix=${data_dir}/relationmat-data/gq
mkdir -p ${data_dir}/relationmat-data

python ./preprocess.py -train_src              ${data_dir}/training_concept_bpe \
                       -train_tgt               ${data_dir}/training_target_bpe \
                       -train_structure1        ${data_dir}/training_0hop_path_bpe  \
                       -train_structure2        ${data_dir}/training_1hop_path_bpe  \
                       -train_structure3        ${data_dir}/training_2hop_path_bpe  \
                       -train_structure4        ${data_dir}/training_3hop_path_bpe  \
                       -train_structure5        ${data_dir}/training_4hop_path_bpe  \
                       -train_structure6        ${data_dir}/training_5hop_path_bpe  \
                       -train_structure7        ${data_dir}/training_6hop_path_bpe  \
                       -train_structure8        ${data_dir}/training_7hop_path_bpe  \
                       -valid_src               ${data_dir}/dev_concept_bpe  \
                       -valid_tgt               ${data_dir}/dev_target_bpe \
                       -valid_structure1        ${data_dir}/dev_0hop_path_bpe   \
                       -valid_structure2        ${data_dir}/dev_1hop_path_bpe   \
                       -valid_structure3        ${data_dir}/dev_2hop_path_bpe   \
                       -valid_structure4        ${data_dir}/dev_3hop_path_bpe   \
                       -valid_structure5        ${data_dir}/dev_4hop_path_bpe   \
                       -valid_structure6        ${data_dir}/dev_5hop_path_bpe   \
                       -valid_structure7        ${data_dir}/dev_6hop_path_bpe   \
                       -valid_structure8        ${data_dir}/dev_7hop_path_bpe   \
                       -save_data ${data_prefix} \
                       -src_vocab_size 20000  \
                       -tgt_vocab_size 20000 \
                       -structure_vocab_size 20000 \
                       -src_seq_length 10000 \
                       -tgt_seq_length 10000 \
                       -shard_size 5000000 \
                       -share_vocab





