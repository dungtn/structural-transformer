#!/bin/bash

data_dir=/mnt/nfs/scratch1/dthai/amred-qa/data/$1/data
mkdir -p ${data_dir}/features-data
data_prefix=${data_dir}/features-data/gq

python preprocess.py -train_src $data_dir/training_concept_bpe \
                        -train_tgt $data_dir/training_target_bpe \
                        -train_structure  $data_dir/training_all_8_path_bpe  \
                        -valid_src $data_dir/dev_concept_bpe  \
                        -valid_tgt $data_dir/dev_target_bpe \
                        -valid_structure $data_dir/dev_all_8_path_bpe   \
                        -save_data $data_prefix \
                        -src_vocab_size 20000  \
                        -tgt_vocab_size 20000 \
                        -structure_vocab_size 20000 \
                        -src_seq_length 20000 \
                        -tgt_seq_length 20000 \
                        -share_vocab





