# -*- coding: UTF-8 -*-
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import MarginRankingLoss
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW
from pysenal import get_logger, write_json
from wsdm_digg.constants import *
from wsdm_digg.reranking.dataloader import RerankDataLoader
from wsdm_digg.reranking.model_loader import load_rerank_model, get_score_func
from wsdm_digg.reranking.parse_args import parse_args
from wsdm_digg.reranking.predict import PlmRerankReranker
from wsdm_digg.benchmark.evaluator import Evaluator


class PlmTrainer(object):
    def __init__(self):
        self.args = parse_args()
        self.plm_model_name = self.args.plm_model_name
        self.rerank_model_name = self.args.rerank_model_name
        self.model_info = MODEL_DICT[self.plm_model_name]
        if 'path' in self.model_info:
            tokenizer_path = self.model_info['path'] + 'vocab.txt'
        else:
            tokenizer_path = self.plm_model_name
        self.tokenizer = self.model_info['tokenizer_class'].from_pretrained(tokenizer_path)
        dest_dir = self.args.dest_base_dir
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        self.train_loader = RerankDataLoader(self.args.train_filename,
                                             self.tokenizer,
                                             self.args,
                                             'train')
        self.logger = get_logger('rerank_trainer')
        logdir = self.args.dest_base_dir + 'logs/'
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)

    def train(self):
        try:
            self.train_func()
        except Exception as e:
            raise e
        finally:
            del self.train_loader

    def train_func(self):
        step = 0
        plm_lr = self.args.plm_learning_rate
        rerank_lr = self.args.rank_learning_rate
        model = load_rerank_model(self.args)
        true_score_func = get_score_func(model, 'true', inference=False)
        false_score_func = get_score_func(model, 'false', inference=False)
        if torch.cuda.is_available():
            model.cuda()
        loss_fct = MarginRankingLoss(margin=1, reduction='mean')

        if self.args.separate_learning_rate:
            params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
            non_bert_params = {'params': [v for k, v in params if not k.startswith('plm_model.')]}
            bert_params = {'params': [v for k, v in params if k.startswith('plm_model.')],
                           'lr': plm_lr}
            # optimizer = torch.optim.Adam([bert_params, non_bert_params], lr=rerank_lr)
            optimizer = AdamW([non_bert_params, bert_params], lr=rerank_lr)
        else:
            optimizer = AdamW(model.parameters(), plm_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_step,
                                              gamma=self.args.scheduler_gamma)
        accumulate_step = 0

        for epoch in range(1, self.args.epoch + 1):
            for batch in self.train_loader:
                model.train()
                true_scores = true_score_func(batch)
                false_scores = false_score_func(batch)
                # y all 1s to indicate positive should be higher
                y = torch.ones(len(true_scores)).float()
                if torch.cuda.is_available():
                    y = y.cuda()

                loss = loss_fct(true_scores, false_scores, y)
                loss.backward()
                self.writer.add_scalar('loss', loss, step)
                accumulate_step += 1

                # torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
                # stop_scheduler_step = self.args.scheduler_step * 8
                if accumulate_step % self.args.gradient_accumulate_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    # if self.args.scheduler_lr and step <= stop_scheduler_step:
                    if self.args.scheduler_lr:# and step <= stop_scheduler_step:
                        scheduler.step()
                    accumulate_step = 0

                step += 1
                if step % self.args.save_model_step == 0:
                    model_basename = self.args.dest_base_dir + self.args.exp_name
                    model_basename += '_epoch_{}_step_{}'.format(epoch, step)
                    torch.save(model.state_dict(), model_basename + '.model')
                    write_json(model_basename + '.json', vars(self.args))
                    map_top3 = self.evaluate(model, 5, model_basename)
                    self.writer.add_scalar('map@3', map_top3, step)
                    self.logger.info('step {} map@3 {:.4f}'.format(step, map_top3))

    def evaluate(self, model, batch_size, mode_basename):
        model_info = {'model': model, 'config': self.args, 'tokenizer': self.tokenizer}
        dest_filename = mode_basename + '_pred_test.jsonl'
        reranker = PlmRerankReranker(model_info, batch_size)
        torch.cuda.empty_cache()
        reranker.rerank_file(search_filename=self.args.search_filename,
                             golden_filename=self.args.test_filename,
                             dest_filename=dest_filename,
                             topk=10, is_submit=False)
        torch.cuda.empty_cache()
        map_val = Evaluator().evaluation_map(dest_filename)
        return map_val


if __name__ == '__main__':
    PlmTrainer().train()
