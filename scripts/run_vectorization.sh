#!/bin/bash

DATA_DIR=${PWD}/data
# PLM_MODEL_NAME='scibert-scivocab-uncased'

# vectorize paper

export CUDA_VISIBLE_DEVICES=1

MODEL_PATH=$DATA_DIR/vectorization/dssm_loss_cls/dssm_loss_cls_epoch_1_step_50000.model
SRC_FILENAME=$DATA_DIR/candidate_paper_for_wsdm2020.jsonl
DEST_FILENAME=$DATA_DIR/candidate_paper_dssm_loss_cls_vector.txt
BATCH_SIZE=20

python3 wsdm_digg/vectorization/predict.py -model_path $MODEL_PATH \
  -src_filename $SRC_FILENAME -dest_filename $DEST_FILENAME \
  -batch_size $BATCH_SIZE -data_type doc

# vectorize validation data
MODEL_PATH=$DATA_DIR/vectorization/dssm_loss_cls/dssm_loss_cls_epoch_1_step_50000.model
SRC_FILENAME=$DATA_DIR/test.jsonl
DEST_FILENAME=$DATA_DIR/test_dssm_loss_cls_vector.txt
BATCH_SIZE=20

python3 wsdm_digg/vectorization/predict.py -model_path $MODEL_PATH \
  -src_filename $SRC_FILENAME -dest_filename $DEST_FILENAME \
  -batch_size $BATCH_SIZE -data_type 'query' -query_field 'description_text'

# vectorize paper

#SRC_FILENAME=$DATA_DIR/candidate_paper_for_wsdm2020.jsonl
#DEST_FILENAME=$DATA_DIR/candidate_paper_scibert_vector.txt
#
#export CUDA_VISIBLE_DEVICES=1
#
#python3 wsdm_digg/vectorization/plm_vectorization.py -plm_model_name $PLM_MODEL_NAME \
#  -src_filename $SRC_FILENAME -dest_filename $DEST_FILENAME

# export CUDA_VISIBLE_DEVICES=0

# # vectorize validation data
# SRC_FILENAME=$DATA_DIR/test.jsonl
# DEST_FILENAME=$DATA_DIR/test_desc_vector.txt
# #echo $SRC_FILENAME
# #echo $DEST_FILENAME
# python3 wsdm_digg/vectorization/plm_vectorization.py -plm_model_name $PLM_MODEL_NAME \
#   -src_filename $SRC_FILENAME -dest_filename $DEST_FILENAME -mode 'query' \
#   -query_field 'description_text'

# #SRC_FILENAME=$DATA_DIR/test.jsonl
# DEST_FILENAME=$DATA_DIR/test_cite_vector.txt

# python3 wsdm_digg/vectorization/plm_vectorization.py -plm_model_name $PLM_MODEL_NAME \
#   -src_filename $SRC_FILENAME -dest_filename $DEST_FILENAME -mode 'query' \
#   -query_field 'cites_text'
