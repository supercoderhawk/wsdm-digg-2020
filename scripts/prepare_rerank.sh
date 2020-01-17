#!/bin/bash

# build data
DATA_DIR=${PWD}/data
SEARCH_FILENAME=$DATA_DIR/result/only_TA_train.jsonl
GOLDEN_FILENAME=$DATA_DIR/train.jsonl
DEST_FILENAME=$DATA_DIR/only_TA_search_result.jsonl

# execute BM25 search to generate candidates for building training data
python3 wsdm_digg/benchmark/benchmarker.py -src_filename $DATA_DIR/train.jsonl \
      -dest_filename $SEARCH_FILENAME

# build training data for reranking
python wsdm_digg/data_process/rerank_data_builder.py -search_filename $SEARCH_FILENAME \
  -golden_filename $GOLDEN_FILENAME -dest_filename $DEST_FILENAME \
  -select_strategy 'search_result_offset' -offset 20 -sample_count 1

DEST_FILENAME=$DATA_DIR/only_TA_sample8.jsonl

python wsdm_digg/data_process/rerank_data_builder.py -search_filename $SEARCH_FILENAME \
  -golden_filename $GOLDEN_FILENAME -dest_filename $DEST_FILENAME \
  -select_strategy 'search_result_offset' -offset 2 -sample_count 8
