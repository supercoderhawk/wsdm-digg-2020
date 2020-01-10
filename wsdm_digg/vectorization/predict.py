# -*- coding: UTF-8 -*-
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from munch import Munch
from pysenal import read_jsonline_lazy,read_json
from wsdm_digg.utils import load_plm_model
from wsdm_digg.vectorization.dataloader import VectorizationDataLoader
from wsdm_digg.vectorization.parse_args import parse_args


class PlmVectorizationPredictor(object):
    def __init__(self, model, tokenizer, args):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        if self.args.data_type == 'query':
            self.id_field = 'description_id'
        else:
            self.id_field = 'paper_id'

    def predict(self, src_filename, dest_filename):
        self.model.eval()
        existed_ids = set()
        for item in read_jsonline_lazy(dest_filename, default=[]):
            existed_ids.add(item[self.id_field])

        loader = VectorizationDataLoader(src_filename, self.tokenizer, self.args, )
        cos = nn.CosineSimilarity(dim=1)
        total_true_tp_count = 0
        total_true_fp_count = 0
        total_false_tp_count = 0
        total_false_fp_count = 0
        total_count = 0
        for batch in loader:
            with torch.no_grad():
                query_embed = self.model(batch, 'query')
                true_embed = self.model(batch, 'true')
                false_embed = self.model(batch, 'false')
                true_scores = cos(query_embed, true_embed)
                false_scores = cos(query_embed, false_embed)
                total_count += query_embed.size(0) * 2
        ret = {'precision': 0, 'recall': 0}
        return ret

    # def load_model(self, model, model_path):
    #     if torch.cuda.is_available():
    #         checkpoint = torch.load(model_path)
    #     else:
    #         checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    #     state_dict = OrderedDict()
    #     # avoid error when load parallel trained model
    #     for k, v in checkpoint.items():
    #         if k.startswith('module.'):
    #             k = k[7:]
    #         state_dict[k] = v
    #     model.load_state_dict(state_dict)
    #     if torch.cuda.is_available():
    #         model = model.cuda()
    #     return model

    # def load_config(self):
    #     # default_config = vars(parse_args(parser=self.parser))
    #     config_path = os.path.splitext(self.model_path)[0] + '.json'
    #     model_config = read_json(config_path)
    #     config_dict = model_config
    #     # config_dict = {**default_config, **model_config}
    #     config_dict['batch_size'] = self.batch_size
    #     config = Munch(config_dict)
    #     return config
