# -*- coding: UTF-8 -*-
import os
from transformers import (BertModel, BertTokenizer,
                          XLNetModel, XLNetTokenizer,
                          RobertaModel, RobertaTokenizer)

DATA_DIR = os.path.abspath(os.path.dirname(__file__) + '/../data/') + '/'
CANDIDATE_FILENAME = DATA_DIR + 'candidate_paper_for_wsdm2020.jsonl'
CANDIDATE_CSV_FILENAME = DATA_DIR + 'candidate_paper_for_wsdm2020.csv'
ES_BASE_URL = 'http://192.168.3.131:9200'
ES_API_URL = ES_BASE_URL + '/wsdm_digg'
RESULT_DIR = DATA_DIR + 'result/'
SUBMIT_DIR = DATA_DIR + 'submit_result/'

BERT_BASE_UNCASED = 'bert-base-uncased'
XLNET_BASE_CASED = 'xlnet-base-cased'
ROBERTA_BASE = 'roberta-base'
SCIBERT_UNCASED = 'scibert-scivocab-uncased'

MODEL_DICT = {BERT_BASE_UNCASED: {'model_class': BertModel,
                                  'tokenizer_class': BertTokenizer},
              XLNET_BASE_CASED: {'model_class': XLNetModel,
                                 'tokenizer_class': XLNetTokenizer},
              ROBERTA_BASE: {'model_class': RobertaModel,
                             'tokenizer_class': RobertaTokenizer},
              SCIBERT_UNCASED: {'model_class': BertModel,
                                'tokenizer_class': BertTokenizer,
                                'path': DATA_DIR + 'scibert_scivocab_uncased/'}
              }
