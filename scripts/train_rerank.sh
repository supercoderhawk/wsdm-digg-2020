#!/bin/bash

EXP_NAME='only_TA_sample8'
DATA_DIR=${PWD}/data
#TRAINING_FILENAME=$DATA_DIR/cite_textrank_top10_rerank_random.jsonl
#TRAINING_FILENAME=$DATA_DIR/cite_textrank_top10_rerank_search_result.jsonl
#TRAINING_FILENAME=$DATA_DIR/only_TA_search_result.jsonl
TRAINING_FILENAME=$DATA_DIR/only_TA_sample8.jsonl
#TRAINING_FILENAME=$DATA_DIR/cite_textrank_top10_rerank_search_result_false_top.jsonl
TEST_FILENAME=$DATA_DIR/test.jsonl

#PLM_MODEL_NAME='bert-base-uncased'
#PLM_MODEL_NAME='xlnet-base-cased'
#PLM_MODEL_NAME='roberta-base'
PLM_MODEL_NAME='scibert-scivocab-uncased'
RERANK_MODEL_NAME='plm'
#RERANK_MODEL_NAME='knrm'
#RERANK_MODEL_NAME='conv-knrm'

DEST_DIR=$DATA_DIR/rerank/${EXP_NAME}_${RERANK_MODEL_NAME}/
#DEST_DIR=$DATA_DIR/rerank/${EXP_NAME}_${RERANK_MODEL_NAME}_context/

export CUDA_VISIBLE_DEVICES=0

python3 wsdm_digg/reranking/trainer.py -exp_name $EXP_NAME \
  -train_filename $TRAINING_FILENAME -test_filename $TEST_FILENAME \
  -search_filename $DATA_DIR/result/only_TA.jsonl \
  -dest_base_dir $DEST_DIR -plm_model_name $PLM_MODEL_NAME \
  -rerank_model_name $RERANK_MODEL_NAME -save_model_step 5000 \
  -mean_list 0.9 0.7 0.5 0.3 0.1 -0.1 -0.3 -0.5 -0.7 -0.9 \
  -stddev_list 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 \
  -batch_size 4 -window_size_list 1 2 3 -gradient_accumulate_step 4 \
  -scheduler_lr -scheduler_step 2500 -scheduler_gamma 0.5
