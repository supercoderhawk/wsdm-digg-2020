# -*- coding: UTF-8 -*-
import os
import json
import torch
import torch.nn as nn
from torch.nn import MarginRankingLoss
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from pysenal import write_json, get_logger
from transformers.optimization import AdamW
from wsdm_digg.vectorization.plm_vectorization import PlmModel
from wsdm_digg.vectorization.predict import PlmVectorizationPredictor
from wsdm_digg.vectorization.dataloader import VectorizationDataLoader
from wsdm_digg.vectorization.parse_args import parse_args


class PlmVectorizationTrainer(object):
    def __init__(self):
        self.args = parse_args()
        self.model = PlmModel(self.args)
        self.tokenizer = self.model.tokenizer
        self.loader = VectorizationDataLoader(self.args.train_filename,
                                              self.model.tokenizer,
                                              args=self.args)
        self.logger = get_logger('vectorization trainer')
        if not os.path.exists(self.args.dest_base_dir):
            os.mkdir(self.args.dest_base_dir)
        self.logdir = self.args.dest_base_dir + '/logs'
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)
        self.writer = SummaryWriter(self.logdir)

    def train(self):
        try:
            self.train_func()
        except Exception as e:
            print(e)
        finally:
            del self.loader

    def train_func(self):
        loss_fct = MarginRankingLoss(margin=2, reduction='mean')
        optimizer = AdamW(self.model.parameters(), self.args.learning_rate)
        step = 0
        cos = nn.CosineSimilarity(dim=1)
        for epoch in range(1, self.args.epoch + 1):
            for batch in self.loader:
                query_embeds = self.model(batch, 'query')
                true_doc_embeds = self.model(batch, 'true')
                false_doc_embeds = self.model(batch, 'false')
                true_scores = cos(query_embeds, true_doc_embeds)
                false_scores = cos(query_embeds, false_doc_embeds)

                y = torch.ones(true_scores.size(0)).float()
                if torch.cuda.is_available():
                    y = y.cuda()
                loss = loss_fct(true_scores, false_scores, y)
                self.writer.add_scalar('loss', loss, step)

                loss.backward()
                optimizer.step()
                step += 1
                if step % self.args.save_model_step == 0:
                    model_basename = self.args.dest_base_dir + self.args.exp_name
                    model_basename += '_epoch_{}_step_{}'.format(epoch, step)
                    torch.save(self.model.state_dict(), model_basename + '.model')
                    write_json(model_basename + '.json', vars(self.args))
                    ret = self.evaluate(model_basename, step)
                    msg_tmpl = 'step {} completed, result {}'
                    self.logger.info(msg_tmpl.format(step, json.dumps(ret)))

    def evaluate(self, model_basename, step):
        torch.cuda.empty_cache()
        predictor = PlmVectorizationPredictor(self.model, self.tokenizer, self.args)
        ret = predictor.predict(self.args.test_filename, model_basename + '_test.jsonl')
        torch.cuda.empty_cache()
        return ret


if __name__ == "__main__":
    PlmVectorizationTrainer().train()
