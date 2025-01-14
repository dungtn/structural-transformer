#!/bin/bash

data_prefix=/mnt/nfs/scratch1/dthai/amred-qa/data/$1/data/relationmat-data/gq
model_dir=/mnt/nfs/scratch1/dthai/amred-qa/models/$1/cnn
mkdir -p ${model_dir}

python train.py \
    -data ${data_prefix} \
    -save_model ${model_dir} \
    -world_size 1  \
    -gpu_ranks 0  \
    -save_checkpoint_steps 5000 \
    -valid_steps 5000 \
    -report_every 20 \
    -keep_checkpoint 30 \
    -seed 3435 \
    -train_steps 300000 \
    -warmup_steps 16000 \
    --share_decoder_embeddings \
    -share_embeddings \
    --position_encoding \
    --optim adam \
    -adam_beta1 0.9 \
    -adam_beta2 0.98 \
    -decay_method noam \
    -learning_rate 0.5 \
    -max_grad_norm 0.0 \
    -batch_size 2048 \
    -batch_type tokens \
    -normalization tokens \
    -dropout 0.1 \
    -label_smoothing 0.1 \
    -max_generator_batches 100 \
    -param_init 0.0 \
    -param_init_glorot \
    -valid_batch_size 8



