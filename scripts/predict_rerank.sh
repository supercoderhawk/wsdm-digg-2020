#!/bin/bash

DATA_DIR=${PWD}/data
RESULT_DIR=$DATA_DIR/submit_result
VALID_FILE=$DATA_DIR/test_release.jsonl
MODEL_BASEDIR=$DATA_DIR/rerank

export CUDA_VISIBLE_DEVICES=0

EPOCH=5
STEP=60000
TOPK=20
MODEL_BASENAME=only_TA_shuffle
RERANK_NAME='plm'

#EPOCH=2
#STEP=130000
#TOPK=20
#MODEL_BASENAME=only_TA_sample8_stop_scheduler
#RERANK_NAME='plm'

MODEL_DIR=$MODEL_BASEDIR/${MODEL_BASENAME}_$RERANK_NAME


python3 wsdm_digg/reranking/predict.py -eval_search_filename $RESULT_DIR/only_TA_release.jsonl \
  -golden_filename $VALID_FILE \
  -dest_filename $RESULT_DIR/${MODEL_BASENAME}_step_${STEP}_top${TOPK}_release.jsonl \
  -model_path $MODEL_DIR/${MODEL_BASENAME}_epoch_${EPOCH}_step_$STEP.model \
  -eval_batch_size 10 -topk $TOPK

EPOCH=5
STEP=60000
TOPK=200
MODEL_BASENAME=only_TA_shuffle
RERANK_NAME='plm'

#EPOCH=2
#STEP=130000
#TOPK=20
#MODEL_BASENAME=only_TA_sample8_stop_scheduler
#RERANK_NAME='plm'

MODEL_DIR=$MODEL_BASEDIR/${MODEL_BASENAME}_$RERANK_NAME


python3 wsdm_digg/reranking/predict.py -eval_search_filename $RESULT_DIR/only_TA_release.jsonl \
  -golden_filename $VALID_FILE \
  -dest_filename $RESULT_DIR/${MODEL_BASENAME}_step_${STEP}_top${TOPK}_release.jsonl \
  -model_path $MODEL_DIR/${MODEL_BASENAME}_epoch_${EPOCH}_step_$STEP.model \
  -eval_batch_size 10 -topk $TOPK
#EPOCH=1
#STEP=80000
#TOPK=20
#MODEL_BASENAME=only_TA_sample8_stop_scheduler
#RERANK_NAME='plm'
#
#MODEL_DIR=$MODEL_BASEDIR/${MODEL_BASENAME}_$RERANK_NAME
#
#
#python3 wsdm_digg/reranking/predict.py -eval_search_filename $RESULT_DIR/only_TA.jsonl \
#  -golden_filename $VALID_FILE \
#  -dest_filename $RESULT_DIR/${MODEL_BASENAME}_step_${STEP}_top${TOPK}.jsonl \
#  -model_path $MODEL_DIR/${MODEL_BASENAME}_epoch_${EPOCH}_step_$STEP.model \
#  -eval_batch_size 10 -topk $TOPK
#
#
#EPOCH=1
#STEP=45000
#TOPK=20
#MODEL_BASENAME=only_TA_sample8_stop_scheduler
#RERANK_NAME='plm'
#
#MODEL_DIR=$MODEL_BASEDIR/${MODEL_BASENAME}_$RERANK_NAME
#
#
#python3 wsdm_digg/reranking/predict.py -eval_search_filename $RESULT_DIR/only_TA.jsonl \
#  -golden_filename $VALID_FILE \
#  -dest_filename $RESULT_DIR/${MODEL_BASENAME}_step_${STEP}_top${TOPK}.jsonl \
#  -model_path $MODEL_DIR/${MODEL_BASENAME}_epoch_${EPOCH}_step_$STEP.model \
#  -eval_batch_size 10 -topk $TOPK
#
#EPOCH=2
#STEP=130000
#TOPK=50
#MODEL_BASENAME=only_TA_sample8_stop_scheduler
#RERANK_NAME='plm'
#
#MODEL_DIR=$MODEL_BASEDIR/${MODEL_BASENAME}_$RERANK_NAME
#
#
#python3 wsdm_digg/reranking/predict.py -eval_search_filename $RESULT_DIR/only_TA.jsonl \
#  -golden_filename $VALID_FILE \
#  -dest_filename $RESULT_DIR/${MODEL_BASENAME}_step_${STEP}_top${TOPK}.jsonl \
#  -model_path $MODEL_DIR/${MODEL_BASENAME}_epoch_${EPOCH}_step_$STEP.model \
#  -eval_batch_size 10 -topk $TOPK