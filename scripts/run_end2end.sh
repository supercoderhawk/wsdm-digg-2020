#!/bin/bash

DATA_DIR=${PWD}/data/
ES_RESULT_FILE=$DATA_DIR/validation_es_result.jsonl
FINAL_RESULT_FILENAME=$DATA_DIR/validation_final_result.jsonl
MODEL_PATH=$DATA_DIR/models/rerank_model.model
TOPK=20

# run elasticsearch (BM25)
python3 wsdm_digg/benchmark/benchmarker.py -src_filename $DATA_DIR/validation.jsonl \
      -dest_filename $ES_RESULT_FILE

# run rerank by bert
python3 wsdm_digg/reranking/predict.py -eval_search_filename $ES_RESULT_FILE \
  -golden_filename $VALID_FILE \
  -dest_filename $RESULT_DIR/$FINAL_RESULT_FILENAME \
  -model_path $MODEL_PATH \
  -eval_batch_size 10 -topk $TOPK