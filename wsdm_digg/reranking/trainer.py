# -*- coding: UTF-8 -*-
import argparse
import torch
from torch.nn import MarginRankingLoss
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW
from pysenal import get_logger, write_json
from wsdm_digg.constants import *
from wsdm_digg.reranking.dataloader import RerankDataLoader
from wsdm_digg.reranking.model_loader import load_model, get_score_func


class PlmTrainer(object):
    def __init__(self):
        self.args = self.parse_args()
        self.plm_model_name = self.args.plm_model_name
        self.rerank_model_name = self.args.rerank_model_name
        self.model_info = MODEL_DICT[self.plm_model_name]
        if 'path' in self.model_info:
            tokenizer_path = self.model_info['path'] + 'vocab.txt'
        else:
            tokenizer_path = self.plm_model_name
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
        rerank_lr = 1e-3
        model = load_model(self.args)
        true_score_func = get_score_func(model, 'true')
        false_score_func = get_score_func(model, 'false')
        if torch.cuda.is_available():
            model.cuda()
        loss_fct = MarginRankingLoss(margin=1, reduction='mean')

        params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
        non_bert_params = {'params': [v for k, v in params if not k.startswith('plm_model.')]}
        bert_params = {'params': [v for k, v in params if k.startswith('plm_model.')], 'lr': bert_lr}
        optimizer = AdamW([non_bert_params, bert_params], lr=rerank_lr)
        for epoch in range(1, self.args.epoch + 1):
            for batch in self.train_loader:
                model.train()
                optimizer.zero_grad()
                true_scores = true_score_func(batch)
                false_scores = false_score_func(batch)
                # y all 1s to indicate positive should be higher
                y = torch.ones(len(true_scores)).float()
                if torch.cuda.is_available():
                    y = y.cuda()
                loss = loss_fct(true_scores, false_scores, y)
                # print(loss)
                self.writer.add_scalar('loss', loss, step)
                loss.backward()

                # if self.args.max_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
                # if self.args.grad_norm:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

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

        parser.add_argument("-epoch", type=int, default=10, help='')
        parser.add_argument("-plm_learning_rate", type=float, default=1e-5, help='')
        parser.add_argument("-ranker_learning_rate", type=float, default=1e-3, help='')
        parser.add_argument("-save_model_step", type=int, default=2000, help='')

        # model parameter
        parser.add_argument("-plm_model_name", required=True, type=str, help='')
        parser.add_argument("-rerank_model_name", required=True, type=str,
                            choices=['plm', 'knrm', 'conv-knrm'], help='')
        parser.add_argument("-max_len", type=int, default=512, help='')
        parser.add_argument("-dim_size", type=int, default=768, help='')
        parser.add_argument("-query_max_len", type=int, default=100, help='')
        parser.add_argument("-special_token_count", type=int, default=2, choices=[2, 3], help='')
        parser.add_argument("-use_context_vector", action='store_true', help='')
        parser.add_argument("-context_merge_method", type=str,
                            choices=['vector_concat', 'score_add'], help='')
        parser.add_argument("-mean_list", type=float, metavar='N', nargs='+', help='')
        parser.add_argument("-stddev_list", type=float, metavar='N', nargs='+', help='')
        parser.add_argument("-window_size_list", type=int, metavar='N', nargs='+', help='')
        parser.add_argument("-filter_size", type=int, default=128, help='')

        args = parser.parse_args(args)
        return args


if __name__ == '__main__':
    PlmTrainer().train()
