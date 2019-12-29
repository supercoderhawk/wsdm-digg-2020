#!/bin/bash

EXP_NAME='bert_pairwise'
DATA_DIR=${PWD}/data/
TRAINING_FILENAME=$DATA_DIR/train.jsonl
TEST_FILENAME=$DATA_DIR/test.jsonl
DEST_DIR=$DATA_DIR/rerank/
MODEL_NAME='bert-base-uncased'

export CUDA_VISIBLE_DEVICES=

python3 wsdm_digg/reranking/trainer.py -exp_name $EXP_NAME \
  -train_filename $TRAINING_FILENAME -test_filename $TEST_FILENAME \
  -dest_base_dir $DEST_DIR -model_name $MODEL_NAME
