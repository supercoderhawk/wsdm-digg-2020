# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from wsdm_digg.constants import *


class PlmRerank(nn.Module):
    model_name = 'plm'

    def __init__(self, args):
        super().__init__()
        self.args = args
        plm_model_name = self.args.plm_model_name
        if plm_model_name not in MODEL_DICT:
            raise ValueError('model name is not supported.')
        model_info = MODEL_DICT[plm_model_name]
        if 'path' in model_info:
            plm_model_name = model_info['path']
        self.plm_model = model_info['model_class'].from_pretrained(plm_model_name)
        if torch.cuda.is_available():
            self.plm_model.cuda()
        self.score_proj = nn.Linear(self.args.dim_size, 1)

    def forward(self, token_ids, segment_ids, token_mask, query_lens, doc_lens):
        output = self.plm_model(input_ids=token_ids,
                                attention_mask=token_mask,
                                token_type_ids=segment_ids)[0]
        scores = torch.tanh(self.score_proj(output[:, 0]))
        return scores
