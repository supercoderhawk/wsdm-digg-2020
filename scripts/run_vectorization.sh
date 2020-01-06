#!/bin/bash

DATA_DIR=${PWD}/data
PLM_MODEL_NAME='scibert-scivocab-uncased'
SRC_FILENAME=$DATA_DIR/candidate_paper_for_wsdm2020.jsonl
DEST_FILENAME=$DATA_DIR/candidate_paper_scibert_vector.h5

export CUDA_VISIBLE_DEVICES=1

python3 wsdm_digg/vectorization/plm.py -plm_model_name $PLM_MODEL_NAME \
  -src_filename $SRC_FILENAME -dest_filename $DEST_FILENAME