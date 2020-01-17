#!/bin/bash

DATA_DIR=${PWD}/data
GOLDEN_FILENAME=$DATA_DIR/test_release.jsonl
ES_RESULT_FILE=$DATA_DIR/test_es_result.jsonl
FINAL_RESULT_FILENAME=$DATA_DIR/test_final_result.jsonl
MODEL_PATH=$DATA_DIR/rerank_model.model
TOPK=50

export CUDA_VISIBLE_DEVICES=1

# run elasticsearch (BM25)
python3 wsdm_digg/benchmark/benchmarker.py -src_filename $DATA_DIR/test_release.jsonl \
      -dest_filename $ES_RESULT_FILE

# run rerank by bert
python3 wsdm_digg/reranking/predict.py -eval_search_filename $ES_RESULT_FILE \
  -golden_filename $GOLDEN_FILENAME \
  -dest_filename $RESULT_DIR/$FINAL_RESULT_FILENAME \
  -model_path $MODEL_PATH \
  -eval_batch_size 10 -topk $TOPK