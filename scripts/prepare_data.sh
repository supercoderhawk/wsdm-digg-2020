#!/bin/bash

# transform csv data to jsonl
python3 wsdm_digg/data_process/raw_data_formatter.py
# split train_release.jsonl to train.jsonl and test.jsonl for experiment train and validation
python3 wsdm_digg/data_process/data_split.py

# build elasticsearch indexing
python3 wsdm_digg/elasticsearch/indexer.py

echo 'index building done!'