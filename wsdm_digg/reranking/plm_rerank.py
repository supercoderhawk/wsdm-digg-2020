# -*- coding: UTF-8 -*-
import argparse
import torch
import torch.nn  as nn
from wsdm_digg.constants import *


class PlmRerank(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = self.parse_args()
        model_name = self.args.model_name
        if model_name not in MODEL_DICT:
            raise ValueError('model name is not supported.')
        model_info = MODEL_DICT[model_name]
        self.tokenizer = model_info['tokenizer_class'].from_pretrained(model_name)
        self.model = model_info['model_class'].from_pretrained(model_name)

    def forward(self, x, x_mask):
        self.model()

