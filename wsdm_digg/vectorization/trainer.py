# -*- coding: UTF-8 -*-
import os
import json
import torch
import torch.nn as nn
from torch.nn import MarginRankingLoss, NLLLoss
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
        if torch.cuda.is_available():
            self.model.cuda()
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
        # self.train_func()
        try:
            self.train_func()
        except Exception as e:
            print(e)
        finally:
            del self.loader

    def train_func(self):
        # loss_fct = MarginRankingLoss(margin=1, reduction='mean')
        loss_fct = NLLLoss(reduction='mean')
        optimizer = AdamW(self.model.parameters(), self.args.learning_rate)
        step = 0
        # cos = nn.CosineSimilarity(dim=1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_step,
                                              gamma=self.args.scheduler_gamma)
        accumulate_step = 0

        for epoch in range(1, self.args.epoch + 1):
            for batch in self.loader:
                probs = self.get_probs(batch)
                batch_size = probs.size(0)

                true_idx = torch.zeros(batch_size, dtype=torch.long)
                if torch.cuda.is_available():
                    true_idx = true_idx.cuda()
                loss = loss_fct(probs, true_idx)
                loss.backward()

                self.writer.add_scalar('loss', loss, step)

                stop_scheduler_step = self.args.scheduler_step * 80

                if accumulate_step % self.args.gradient_accumulate_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if self.args.scheduler_lr and step <= stop_scheduler_step:
                        scheduler.step()
                    accumulate_step = 0

                step += 1
                if step % self.args.save_model_step == 0:
                    model_basename = self.args.dest_base_dir + self.args.exp_name
                    model_basename += '_epoch_{}_step_{}'.format(epoch, step)
                    torch.save(self.model.state_dict(), model_basename + '.model')
                    write_json(model_basename + '.json', vars(self.args))
                    ret = self.evaluate(model_basename, step)
                    self.writer.add_scalar('accuracy', ret, step)
                    # self.writer.add_scalar('recall', ret['recall'], step)
                    # self.writer.add_scalar('f1', ret['f1'], step)
                    msg_tmpl = 'step {} completed, accuracy {:.4f}'
                    self.logger.info(msg_tmpl.format(step, ret))

    def get_probs(self, batch):
        cos = nn.CosineSimilarity(dim=1)
        query_embeds = self.model(batch, 'query')
        true_doc_embeds = self.model(batch, 'true')
        true_scores = cos(query_embeds, true_doc_embeds)
        false_doc_embeds = self.model(batch, 'false')

        if not isinstance(false_doc_embeds, list):
            false_scores = cos(query_embeds, false_doc_embeds)
            scores = torch.stack([true_scores, false_scores], dim=1)
        else:
            scores = true_scores.unsqueeze(1)
            for false_single_doc_embed in false_doc_embeds:
                false_scores = cos(query_embeds, false_single_doc_embed)
                scores = torch.cat([scores, false_scores.unsqueeze(1)], dim=1)

        probs = torch.log_softmax(scores, dim=-1)
        return probs

    def evaluate(self, model_basename, step):
        torch.cuda.empty_cache()
        model_info = {'model': self.model,
                      'tokenizer': self.tokenizer,
                      'config': self.args}
        predictor = PlmVectorizationPredictor(model_info)
        ret = predictor.predict(self.args.test_filename, model_basename + '_test.jsonl')
        torch.cuda.empty_cache()
        return ret


if __name__ == "__main__":
    PlmVectorizationTrainer().train()
