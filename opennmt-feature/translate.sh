#!/bin/bash

data_dir=/mnt/nfs/scratch1/dthai/amred-qa/data/$1/data
model_fn=/mnt/nfs/scratch1/dthai/amred-qa/models/$1/features8_step_300000.pt
output_dir=/mnt/nfs/scratch1/dthai/amred-qa/results/$1
mkdir -p ${output_dir}

python translate.py -model ${model_fn} \
 -src ${data_dir}/$2_concept_bpe \
 -structure  ${data_dir}/$2_all_path_bpe \
 -output     ${output_dir}/$2_target_features8.translate \
 -beam_size 5 \
 -share_vocab \
 -gpu 0


