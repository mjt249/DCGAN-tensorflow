#!/bin/bash
# bash script to launch the GAN. Change the input flags and directories if necessary.
# Chiyu "Max" Jiang, 05/06/17

ROOT_DIR=/home/maxjiang/sdfgan
DATASET_DIR=${ROOT_DIR}/data
CHECKPOINT_DIR=${ROOT_DIR}/checkpoint
LOG_DIR=${ROOT_DIR}/logs

python main.py \
  --is_convert_latent \
  --batch_size=64 \
  --classification_dataset=ModelNet10 \
  --dataset=shapenet \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_dir=${CHECKPOINT_DIR} \
  --log_dir=${LOG_DIR} \
  --is_classifier \
  --classifier_epoch=25 \
