# -*- coding: UTF-8 -*-
import argparse
from wsdm_digg.constants import *
from wsdm_digg.reranking.dataloader import RerankDataLoader


class PlmTrainer(object):
    def __init__(self):
        self.args = self.parse_args()
        self.model_name = self.args.model_name
        self.model_info = MODEL_DICT[self.model_name]
        self.tokenizer = self.model_info['tokenizer_class'].from_pretrained(self.model_name)
        self.train_loader = RerankDataLoader(self.args.train_filename,
                                             self.tokenizer,
                                             self.args)

    def train(self):
        step = 0
        model = self.model_info['model_class'].from_pretrained(self.model_name)
        for epoch in range(1, self.args.epoch + 1):
            for batch in self.train_loader:
                model()
                step += 1

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument()

        args = parser.parse_args(args)
        return args


if __name__ == '__main__':
    PlmTrainer().train()
