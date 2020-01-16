#!/bin/bash

EXP_NAME='dssm_loss'
DATA_DIR=${PWD}/data
TRAIN_FILENAME=$DATA_DIR/only_TA_sample10_aggregate.jsonl
TEST_FILENAME=$DATA_DIR/test_vectorization.jsonl
DEST_DIR=$DATA_DIR/vectorization/${EXP_NAME}/

PLM_MODEL_NAME='scibert-scivocab-uncased'

export CUDA_VISIBLE_DEVICES=0

python3 wsdm_digg/vectorization/trainer.py -train_filename $TRAIN_FILENAME \
    -test_filename $TEST_FILENAME -exp_name $EXP_NAME -dest_base_dir $DEST_DIR \
    -mode train -plm_model_name $PLM_MODEL_NAME -embed_mode USE -batch_size 1 \
    -scheduler_lr -gradient_accumulate_step 8 -query_field 'description_text' \
    -learning_rate 1e-5
