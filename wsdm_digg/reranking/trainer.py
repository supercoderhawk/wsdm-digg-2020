# -*- coding: UTF-8 -*-
import argparse
import torch
from torch.nn import MarginRankingLoss
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW
from pysenal import get_logger, write_json
from wsdm_digg.constants import *
from wsdm_digg.reranking.dataloader import RerankDataLoader
from wsdm_digg.reranking.plm_rerank import PlmRerank


class PlmTrainer(object):
    def __init__(self):
        self.args = self.parse_args()
        self.model_name = self.args.model_name
        self.model_info = MODEL_DICT[self.model_name]
        if 'path' in self.model_info:
            tokenizer_path = self.model_info['path'] + 'vocab.txt'
        else:
            tokenizer_path = self.model_name
        self.tokenizer = self.model_info['tokenizer_class'].from_pretrained(tokenizer_path)
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
        bert_lr = 1e-5
        model = PlmRerank(self.args)
        if torch.cuda.is_available():
            model.cuda()
        loss_fct = MarginRankingLoss(margin=1, reduction='mean')
        optimizer = AdamW(model.parameters(), lr=bert_lr)
        for epoch in range(1, self.args.epoch + 1):
            for batch in self.train_loader:
                model.train()
                optimizer.zero_grad()
                pos_scores = model(token_ids=batch['true_token'],
                                   segment_ids=batch['true_segment'],
                                   token_mask=batch['true_mask'])
                neg_scores = model(token_ids=batch['false_token'],
                                   segment_ids=batch['false_segment'],
                                   token_mask=batch['false_mask'])
                # y all 1s to indicate positive should be higher
                y = torch.ones(len(pos_scores)).float()
                if torch.cuda.is_available():
                    y = y.cuda()
                loss = loss_fct(pos_scores, neg_scores, y)
                self.writer.add_scalar('loss', loss, step)
                loss.backward()
                optimizer.step()
                step += 1
                if step % self.args.save_model_step == 0:
                    model_basename = self.args.dest_base_dir + self.args.exp_name
                    model_basename += '_epoch_{}_step_{}'.format(epoch, step)
                    torch.save(model.state_dict(), model_basename + '.model')
                    write_json(model_basename + '.json', vars(self.args))

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("-exp_name", required=True, type=str, help='')
        parser.add_argument("-train_filename", required=True, type=str, help='')
        parser.add_argument("-test_filename", required=True, type=str, help='')
        parser.add_argument("-dest_base_dir", required=True, type=str, help='')
        parser.add_argument("-batch_size", type=int, default=4, help='')
        parser.add_argument("-max_len", type=int, default=512, help='')
        parser.add_argument("-dim_size", type=int, default=768, help='')
        parser.add_argument("-epoch", type=int, default=5, help='')
        parser.add_argument("-save_model_step", type=int, default=2000, help='')

        # model parameter
        parser.add_argument("-model_name", required=True, type=str, help='')

        args = parser.parse_args(args)
        return args


if __name__ == '__main__':
    PlmTrainer().train()
