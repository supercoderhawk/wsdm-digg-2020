#!/bin/bash

DATA_DIR=${PWD}/data
RESULT_DIR=$DATA_DIR/submit_result
VALID_FILE=$DATA_DIR/validation.jsonl
MODEL_BASEDIR=$DATA_DIR/rerank

EPOCH=2
STEP=115000
TOPK=20
MODEL_BASENAME=only_TA_sample8
RERANK_NAME='plm'

MODEL_DIR=$MODEL_BASEDIR/${MODEL_BASENAME}_$RERANK_NAME

export CUDA_VISIBLE_DEVICES=1

python3 wsdm_digg/reranking/predict.py -eval_search_filename $RESULT_DIR/only_TA.jsonl \
  -golden_filename $VALID_FILE \
  -dest_filename $RESULT_DIR/${MODEL_BASENAME}_step_${STEP}_top${TOPK}.jsonl \
  -model_path $MODEL_DIR/${MODEL_BASENAME}_epoch_${EPOCH}_step_$STEP.model \
  -eval_batch_size 10
