#!/bin/bash

DATA_DIR=${PWD}/data/
ES_RESULT_FILE=$DATA_DIR/validation_es_result.jsonl

# run elasticsearch for
python3 wsdm_digg/benchmark/benchmarker.py -src_filename $DATA_DIR/validation.jsonl \
      -dest_filename $ES_RESULT_FILE

# run rerank
python3 wsdm_digg/reranking/predict.py -search_filename $ES_RESULT_FILE \
      -golden_filename $DATA_DIR/validation.jsonl