#!/bin/bash

DATA_DIR=${PWD}/data
PLM_MODEL_NAME='scibert-scivocab-uncased'

# vectorize paper

#SRC_FILENAME=$DATA_DIR/candidate_paper_for_wsdm2020.jsonl
#DEST_FILENAME=$DATA_DIR/candidate_paper_scibert_vector.txt
#
#export CUDA_VISIBLE_DEVICES=1
#
#python3 wsdm_digg/vectorization/plm.py -plm_model_name $PLM_MODEL_NAME \
#  -src_filename $SRC_FILENAME -dest_filename $DEST_FILENAME

export CUDA_VISIBLE_DEVICES=0

# vectorize validation data
SRC_FILENAME=$DATA_DIR/test.jsonl
DEST_FILENAME=$DATA_DIR/test_desc_vector.txt
#echo $SRC_FILENAME
#echo $DEST_FILENAME
python3 wsdm_digg/vectorization/plm.py -plm_model_name $PLM_MODEL_NAME \
  -src_filename $SRC_FILENAME -dest_filename $DEST_FILENAME -mode 'query' \
  -query_field 'description_text'

#SRC_FILENAME=$DATA_DIR/test.jsonl
DEST_FILENAME=$DATA_DIR/test_cite_vector.txt

python3 wsdm_digg/vectorization/plm.py -plm_model_name $PLM_MODEL_NAME \
  -src_filename $SRC_FILENAME -dest_filename $DEST_FILENAME -mode 'query' \
  -query_field 'cites_text'