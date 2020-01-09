# -*- coding: UTF-8 -*-
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pysenal import read_jsonline_lazy, get_chunk, read_lines_lazy, append_lines
from torch.multiprocessing import Queue, Process
from wsdm_digg.constants import MODEL_DICT


class VectorizationDataLoader(object):
    def __init__(self, src_filename, tokenizer, args, existed_ids=None):
        self.args = args
        self.src_filename = src_filename
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.query_field = args.query_field
        self.mode = args.mode
        self.input_queue = Queue(-1)
        self.output_queue = Queue(-1)
        # print(self.existed_ids)
        self.existed_ids = set(existed_ids)
        self.data = self.get_data()
        # print(next(get_chunk(read_jsonline_lazy(self.src_filename), self.batch_size)))
        self.worker_num = 10
        self.workers = []
        self._batch_in_queue = 0
        if self.mode == 'query':
            self.id_field = 'description_id'
        else:
            self.id_field = 'paper_id'
        for _ in range(self.worker_num):
            worker = Process(target=self._worker_loop)
            self.workers.append(worker)

        self.__prefetch()
        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def get_data(self):
        if not self.existed_ids:
            for chunk in get_chunk(read_jsonline_lazy(self.src_filename), self.batch_size):
                yield chunk
            # print(self.existed_ids)
            # return get_chunk(read_jsonline_lazy(self.src_filename), self.batch_size)
        else:
            raw_batch = []
            for item in read_jsonline_lazy(self.src_filename):
                if item[self.id_field] in self.existed_ids:
                    continue
                raw_batch.append(item)
                if len(raw_batch) == self.batch_size:
                    yield raw_batch
                    raw_batch = []
            if raw_batch:
                yield raw_batch

    def _worker_loop(self):
        pad_id = self.tokenizer.pad_token_id
        while True:
            raw_batch = self.input_queue.get()
            if raw_batch is None:
                break
            token_ids_list = []
            data_id_list = []
            mask_ids_list = []
            for item in raw_batch:
                if self.mode == 'doc':
                    text = item['title'] + ' ' + item['abstract']
                elif self.mode == 'query':
                    text = item[self.query_field]
                token_ids = self.tokenizer.encode(text, max_length=self.max_length,
                                                  add_special_tokens=True)
                mask_ids = np.arange(self.max_length) <= len(token_ids)
                pad_len = self.max_length - len(token_ids)
                token_ids.extend([pad_id] * pad_len)
                token_ids_list.append(token_ids)
                data_id_list.append(item[self.id_field])
                mask_ids_list.append(mask_ids)

            batch = {'tokens': np.array(token_ids_list, dtype=np.long),
                     'masks': np.array(mask_ids_list),
                     'data_ids': data_id_list}
            self.output_queue.put(batch)

    def __prefetch(self):
        for batch_idx in range(10):
            raw_batch = next(self.data)
            self.input_queue.put(raw_batch)
            self._batch_in_queue += 1

    def __iter__(self):
        for raw_batch in self.data:
            if self._batch_in_queue:
                batch = self.output_queue.get()
                self.batch2tensor(batch)
                yield batch
                self.input_queue.put(raw_batch)

        for _ in range(self._batch_in_queue):
            batch = self.output_queue.get()
            self.batch2tensor(batch)
            yield batch

    def batch2tensor(self, batch):
        batch['tokens'] = torch.tensor(batch['tokens'])
        batch['masks'] = torch.tensor(batch['masks'])
        if torch.cuda.is_available():
            batch['tokens'] = batch['tokens'].cuda()
            batch['masks'] = batch['masks'].cuda()

    def __del__(self):
        self.input_queue.close()
        self.output_queue.close()
        for worker in self.workers:
            worker.terminate()


class PlmVectorization(object):
    def __init__(self):
        self.args = self.parse_args()
        plm_model_name = self.args.plm_model_name
        if plm_model_name not in MODEL_DICT:
            raise ValueError('model name is not supported.')
        model_info = MODEL_DICT[plm_model_name]
        if 'path' in model_info:
            plm_model_name = model_info['path']
        self.plm_model = model_info['model_class'].from_pretrained(plm_model_name)

        if torch.cuda.is_available():
            self.plm_model.cuda()
        if 'path' in model_info:
            tokenizer_path = model_info['path'] + 'vocab.txt'
        else:
            tokenizer_path = self.args.plm_model_name

        self.tokenizer = model_info['tokenizer_class'].from_pretrained(tokenizer_path)

    def run(self):
        dest_filename = self.args.dest_filename
        existed_data_ids = {l.split()[0] for l in read_lines_lazy(dest_filename, default=[])}
        # print(len(existed_data_ids))
        loader = VectorizationDataLoader(src_filename=self.args.src_filename,
                                         tokenizer=self.tokenizer,
                                         existed_ids=existed_data_ids,
                                         args=self.args)
        lines = []
        self.plm_model.eval()
        for batch in loader:
            with torch.no_grad():
                output = self.plm_model(input_ids=batch['tokens'],
                                        attention_mask=batch['masks'])[0]
            for sent_embed, data_id in zip(output[:, 0], batch['data_ids']):
                if data_id in existed_data_ids:
                    continue
                if not data_id.strip():
                    continue
                if torch.cuda.is_available():
                    sent_embed = sent_embed.cpu()
                sent_embed = sent_embed.detach().numpy()
                vec_str = np.array2string(sent_embed, separator=' ', floatmode='maxprec')[1:-1]
                vec_str = ' '.join([line.strip() for line in vec_str.splitlines(False)])
                line = data_id + ' ' + vec_str
                lines.append(line)
            append_lines(dest_filename, lines)
            lines = []

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-plm_model_name', type=str, required=True, help='')
        parser.add_argument('-src_filename', type=str, required=True, help='')
        parser.add_argument('-dest_filename', type=str, required=True, help='')
        parser.add_argument('-batch_size', type=int, default=10, help='')
        parser.add_argument('-max_length', type=int, default=512, help='')
        parser.add_argument('-mode', type=str, required=True, choices=['query', 'doc'], help='')
        parser.add_argument('-query_field', type=str, default='cites_text',
                            choices=['cites_text', 'description_text'], help='')
        args = parser.parse_args()
        return args


class PlmModel(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()
        plm_model_name = self.args.plm_model_name
        if plm_model_name not in MODEL_DICT:
            raise ValueError('model name is not supported.')
        model_info = MODEL_DICT[plm_model_name]
        if 'path' in model_info:
            plm_model_name = model_info['path']
        self.plm_model = model_info['model_class'].from_pretrained(plm_model_name)

        if torch.cuda.is_available():
            self.plm_model.cuda()
        if 'path' in model_info:
            tokenizer_path = model_info['path'] + 'vocab.txt'
        else:
            tokenizer_path = self.args.plm_model_name

        self.tokenizer = model_info['tokenizer_class'].from_pretrained(tokenizer_path)
        self.attn_proj = nn.Linear(self.args.dim_size, self.args.dim_size)

    def forward(self, batch, prefix):
        if not prefix:
            token_field = 'tokens'
            mask_field = 'masks'
            len_field = 'token_lens'
        else:
            token_field = '{}_tokens'.format(prefix)
            mask_field = '{}_masks'.format(prefix)
            len_field = '{}_token_lens'.format(prefix)

        output = self.plm_model(input_ids=batch[token_field],
                                attention_mask=batch[mask_field])[0]

        cls_embed = output[:, 0]
        token_embed = output[:, 1:]
        token_mask = batch[mask_field][:, 1:].unsqueeze(2)
        sent_lens = batch[len_field]

        if self.args.embed_mode == 'USE':
            sent_embed = self.get_USE_embedding(sent_lens, token_embed, token_mask)
        else:
            sent_embed = self.get_attention_embedding(cls_embed, token_embed, token_mask)

        return sent_embed

    def get_attention_embedding(self, cls_embed, token_embed, token_mask):
        """
        attention Model
        :param cls_embed:
        :param token_embed:
        :param token_mask:
        :return:
        """
        attn_weights = torch.bmm(self.attn_proj(cls_embed.unsqueeze(1)), token_embed)
        attn_weights.masked_fill_(token_mask, float('-inf'))
        attn_scores = torch.softmax(attn_weights, dim=-1)
        context_embed = torch.sum(attn_scores * token_embed, dim=1)
        sent_embed = (cls_embed + context_embed) / 2
        return sent_embed

    def get_USE_embedding(self, sent_lens, token_embed, token_mask):
        """
        Universal Sentence Encoder Transformer Model
        :param sent_lens:
        :param token_embed:
        :param token_mask:
        :return:
        """
        masked_token_embed = token_embed.mask_fill_(token_mask, 0.0)
        sent_embed = masked_token_embed.sum(dim=1) / torch.sqrt(sent_lens)

        return sent_embed


if __name__ == '__main__':
    PlmVectorization().run()
