# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from wsdm_digg.constants import *


class PlmRerank(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model_name = self.args.model_name
        if model_name not in MODEL_DICT:
            raise ValueError('model name is not supported.')
        model_info = MODEL_DICT[model_name]
        self.tokenizer = model_info['tokenizer_class'].from_pretrained(model_name)
        self.model = model_info['model_class'].from_pretrained(model_name)
        self.score_proj = nn.Linear(self.args.dim_size, 1)

    def forward(self, token_ids, segment_ids, token_mask):
        output = self.model(input_ids=token_ids,
                            attention_mask=token_mask,
                            token_type_ids=segment_ids)[0]
        # scores = self.score_proj(output[:, 0])
        scores = self.score_proj(F.tanh(output[:, 0]))
        return scores
