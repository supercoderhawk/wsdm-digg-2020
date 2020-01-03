#!/bin/bash

# build data
DATA_DIR=${PWD}/data
SEARCH_FILENAME=$DATA_DIR/result/cite_textrank_top10_train.jsonl
GOLDEN_FILENAME=$DATA_DIR/train.jsonl
DEST_FILENAME=$DATA_DIR/cite_textrank_top10_rerank_search_result_fasle_top.jsonl

python wsdm_digg/data_process/rerank_data_builder.py -search_filename $SEARCH_FILENAME \
  -golden_filename $GOLDEN_FILENAME -dest_filename $DEST_FILENAME \
  -select_strategy 'search_result_false_top'

