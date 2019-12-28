#!/bin/bash

# transform csv data to jsonl
python3 wsdm_digg/data_process/raw_data_formatter.py
# build elasticsearch indexing
python3 wsdm_digg/elasticsearch/indexer.py