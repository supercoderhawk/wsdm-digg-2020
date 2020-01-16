# -*- coding: UTF-8 -*-
import os
import numpy as np
import argparse
from collections import OrderedDict
from torch.multiprocessing import Queue, Process
import torch
import torch.nn as nn
from munch import Munch
from pysenal import read_jsonline_lazy, read_json, append_lines
from wsdm_digg.vectorization.dataloader import VectorizationDataLoader
from wsdm_digg.vectorization.plm_vectorization import PlmModel


class PlmVectorizationPredictor(object):
    def __init__(self, model_info, config=None, vectorization=False):
        if isinstance(model_info, dict):
            self.args = model_info['config']
            self.model = model_info['model']
            self.tokenizer = model_info['tokenizer']
        elif isinstance(model_info, str):
            self.model_path = model_info
            self.args = self.load_config(config)
            model = PlmModel(self.args)
            self.model = self.load_model(model, model_info)
            self.tokenizer = self.model.tokenizer
        else:
            raise ValueError('error..')
        if self.args.data_type == 'query':
            self.id_field = 'description_id'
        else:
            self.id_field = 'paper_id'

        if vectorization:
            self.dest_filename = self.args.dest_filename
            self.output_queue = Queue(-1)
            self.worker = Process(target=self.np2str)
            self.worker.daemon = True
            self.worker.start()

    def predict(self, src_filename, dest_filename):
        self.model.eval()
        existed_ids = set()
        for item in read_jsonline_lazy(dest_filename, default=[]):
            existed_ids.add(item[self.id_field])

        loader = VectorizationDataLoader(src_filename, self.tokenizer, self.args)
        cos = nn.CosineSimilarity(dim=1)
        tp_count = 0

        total_count = 0
        for batch in loader:
            with torch.no_grad():
                query_embed = self.model(batch, 'query')
                true_embed = self.model(batch, 'true')
                false_embed = self.model(batch, 'false')
                true_scores = cos(query_embed, true_embed)
                false_scores = cos(query_embed, false_embed)
                print(true_scores, false_scores)
                total_count += query_embed.size(0)
                tp_count += (true_scores > false_scores).sum().cpu().numpy().tolist()

        accuray = tp_count / total_count

        return accuray

    def np2str(self):
        while True:
            batch = self.output_queue.get(block=True)
            lines = []
            for sent_embed, data_id in zip(batch['vector'], batch['index']):
                vec_str = np.array2string(sent_embed,
                                          separator=' ', floatmode='maxprec')[1:-1]
                vec_str = ' '.join([line.strip() for line in vec_str.splitlines(False)])
                line = data_id + ' ' + vec_str
                lines.append(line)
            # if not getattr(self,'dest_filename',None):
            #     print(lines, batch)
            append_lines(self.dest_filename, lines)

    def vectorize(self, src_filename, dest_filename):
        self.dest_filename = dest_filename
        loader = VectorizationDataLoader(src_filename, self.tokenizer, self.args)

        for batch in loader:
            with torch.no_grad():
                sent_embed_list = self.model(batch, prefix=None).cpu().numpy()
                self.output_queue.put({'vector': sent_embed_list, 'index': batch['data_ids']})

    def load_model(self, model, model_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = OrderedDict()
        # avoid error when load parallel trained model
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v
        model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def load_config(self, custom_config):
        # default_config = vars(parse_args(parser=self.parser))
        config_path = os.path.splitext(self.model_path)[0] + '.json'
        model_config = read_json(config_path)
        if custom_config:
            config_dict = {**model_config, **custom_config}
        else:
            config_dict = model_config
        config = Munch(config_dict)
        return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str)
    parser.add_argument('-src_filename', type=str)
    parser.add_argument('-dest_filename', type=str)
    parser.add_argument('-batch_size', type=int)
    parser.add_argument('-data_type', type=str, choices=['query', 'doc'], default='doc')
    parser.add_argument('-query_field', type=str, choices=['cites_text', 'description_text'],
                        default='cites_text')
    args = parser.parse_args()
    predictor = PlmVectorizationPredictor(model_info=args.model_path,
                                          config={'data_type': args.data_type,
                                                  'batch_size': args.batch_size,
                                                  'query_field': args.query_field,
                                                  'mode': 'inference',
                                                  'dest_filename': args.dest_filename},
                                          vectorization=True)
    predictor.vectorize(args.src_filename, args.dest_filename)


if __name__ == "__main__":
    main()
