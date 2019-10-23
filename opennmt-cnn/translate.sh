#!/bin/bash

data_dir=/mnt/nfs/scratch1/dthai/amred-qa/data/$1/data
model_fn=/mnt/nfs/scratch1/dthai/amred-qa/models/$1/cnn_step_300000.pt
output_dir=/mnt/nfs/scratch1/dthai/amred-qa/results/$1
mkdir -p ${output_dir}

python translate.py -model ${model_fn} \
 -src ${data_dir}/$2_concept_bpe \
 -structure1  ${data_dir}/$2_0hop_path_bpe \
 -structure2  ${data_dir}/$2_1hop_path_bpe \
 -structure3  ${data_dir}/$2_2hop_path_bpe \
 -structure4  ${data_dir}/$2_3hop_path_bpe \
 -structure5  ${data_dir}/$2_4hop_path_bpe \
 -structure6  ${data_dir}/$2_5hop_path_bpe \
 -structure7  ${data_dir}/$2_6hop_path_bpe \
 -structure8  ${data_dir}/$2_7hop_path_bpe \
 -output     ${output_dir}/$2_target_cnn_bpe.translate \
 -share_vocab \
 -gpu 0

