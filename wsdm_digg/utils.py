# -*- coding: UTF-8 -*-
import os
import torch
from pysenal import read_jsonline_lazy, append_line
from wsdm_digg.constants import MODEL_DICT


def result_format(src_filename, dest_filename=None):
    if dest_filename is None:
        dest_filename = os.path.splitext(src_filename)[0] + '.csv'
    for item in read_jsonline_lazy(src_filename):
        desc_id = item['description_id']
        paper_ids = item['docs'][:3]
        if not paper_ids:
            raise ValueError('result is empty')
        line = desc_id + '\t' + '\t'.join(paper_ids)
        append_line(dest_filename, line)


def load_plm_model(plm_model_name):
    if plm_model_name not in MODEL_DICT:
        raise ValueError('model name is not supported.')
    model_info = MODEL_DICT[plm_model_name]
    if 'path' in model_info:
        plm_model_name = model_info['path']
    plm_model = model_info['model_class'].from_pretrained(plm_model_name)

    if torch.cuda.is_available():
        plm_model.cuda()
    if 'path' in model_info:
        tokenizer_path = model_info['path'] + 'vocab.txt'
    else:
        tokenizer_path = plm_model_name

    tokenizer = model_info['tokenizer_class'].from_pretrained(tokenizer_path)
    return plm_model, tokenizer
