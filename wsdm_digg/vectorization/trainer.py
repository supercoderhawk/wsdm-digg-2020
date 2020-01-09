# -*- coding: UTF-8 -*-
import argparse
import torch
import torch.nn as nn
from torch.nn import MarginRankingLoss
import torch.optim as optim
from transformers.optimization import AdamW
from wsdm_digg.vectorization.plm import PlmModel
from wsdm_digg.vectorization.plm import VectorizationDataLoader


class PlmVectorizationTrainer(object):
    def __init__(self):
        self.args = self.parse_args()
        self.model = PlmModel(self.args)
        self.loader = VectorizationDataLoader(self.args.train_filename,)

    def train(self):
        try:
            self.train_func()
        except Exception as e:
            print(e)
        finally:
            del self.loader

    def train_func(self):
        loss_fct = MarginRankingLoss(margin=1, reduction='mean')
        optimizer = AdamW(self.model.parameters(), self.args.learning_rate)
        step = 0
        cos = nn.CosineSimilarity(dim=1)
        for epoch in range(1, self.args.epoch + 1):
            for batch in self.loader:
                query_embeds = self.model(batch)
                true_doc_embeds = self.model(batch, 'true')
                false_doc_embeds = self.model(batch, 'false')
                true_scores = cos(query_embeds, true_doc_embeds)
                false_scores = cos(query_embeds, false_doc_embeds)

                y = torch.ones(len(true_scores)).float()
                if torch.cuda.is_available():
                    y = y.cuda()
                loss = loss_fct(true_scores, false_scores, y)
                loss.backward()
                optimizer.step()
                step += 1
                if step % self.args.save_model_step == 0:
                    self.evaluate()

    def evaluate(self):
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-train_filename')
        parser.add_argument('-test_filename')

        args = parser.parse_args()
        return args


if __name__ == "__main__":
    PlmVectorizationTrainer().train()
