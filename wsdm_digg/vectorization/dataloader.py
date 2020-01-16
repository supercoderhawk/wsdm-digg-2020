# -*- coding: UTF-8 -*-
import random
import numpy as np
import torch
from pysenal import read_jsonline_lazy, get_chunk, read_jsonline
from torch.multiprocessing import Queue, Process


class VectorizationDataLoader(object):
    def __init__(self, src_filename, tokenizer, args, existed_ids=None):
        self.src_filename = src_filename
        self.tokenizer = tokenizer
        self.args = args
        self.existed_ids = existed_ids

    def __iter__(self):
        i = VectorizationDataItertor(self.src_filename, self.tokenizer, self.args, self.existed_ids)
        return iter(i)


class VectorizationDataItertor(object):
    def __init__(self, src_filename, tokenizer, args, existed_ids=None):
        self.args = args
        self.src_filename = src_filename
        self.tokenizer = tokenizer
        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.query_field = args.query_field
        self.mode = args.mode
        self.data_type = args.data_type
        self.input_queue = Queue(-1)
        self.output_queue = Queue(-1)
        if existed_ids is not None:
            self.existed_ids = set(existed_ids)
        else:
            self.existed_ids = set()
        self._data = self.get_data()
        self.worker_num = 10
        self.workers = []
        self._batch_in_queue = 0
        if self.data_type == 'query':
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
            if self.args.lazy_loading:
                data = read_jsonline_lazy(self.src_filename)
            else:
                data = read_jsonline(self.src_filename)
                random.shuffle(data)
                random.shuffle(data)
                random.shuffle(data)
            for chunk in get_chunk(data, self.batch_size):
                yield chunk
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
        while True:
            raw_batch = self.input_queue.get()
            if raw_batch is None:
                break
            if self.mode == 'train':
                batch = self.process_train_batch(raw_batch)
            elif self.mode == 'eval':
                batch = self.process_train_batch(raw_batch)
            elif self.mode == 'inference':
                batch = self.process_inference_batch(raw_batch)
            else:
                raise ValueError('mode error.....')
            self.output_queue.put(batch)

    def process_train_batch(self, raw_batch):
        query_batch = self.process_inference_batch(raw_batch, 'query')
        true_batch = self.process_inference_batch(raw_batch, 'true')
        false_batch = self.process_inference_batch(raw_batch, 'false')
        final_batch = {**query_batch, **true_batch, **false_batch}
        return final_batch

    def process_inference_batch(self, raw_batch, prefix=None):
        pad_id = self.tokenizer.pad_token_id
        src_data_id = self.id_field
        if prefix in {'true', 'false'}:
            src_data_id = '{}_paper_id'.format(prefix)
        elif prefix == 'query':
            src_data_id = 'description_id'

        token_ids_list = []
        data_id_list = []
        mask_ids_list = []
        sent_lens_list = []
        is_text_list = False
        for item in raw_batch:
            if prefix:
                if prefix == 'query':
                    if self.query_field == 'cites_text':
                        text = item['query']
                    else:
                        text = item[self.query_field]
                elif prefix == 'true':
                    text = item['true_doc']
                elif prefix == 'false':
                    text = item['false_doc']
                else:
                    raise ValueError('prefix error')
            else:
                if self.data_type == 'doc':
                    text = item['title'] + ' ' + item['abstract']
                elif self.data_type == 'query':
                    if self.query_field == 'cites_text' and self.query_field not in item:
                        text = item['query']
                    else:
                        text = item[self.query_field]
                else:
                    raise ValueError('error')
            if isinstance(text, list):
                text = text[:self.args.negative_sample_count]
                list_size = len(text)
                is_text_list = True
                if not token_ids_list:
                    token_ids_list = [[] for _ in range(list_size)]
                    data_id_list = [[] for _ in range(list_size)]
                    mask_ids_list = [[] for _ in range(list_size)]
                    sent_lens_list = [[] for _ in range(list_size)]
                for text_idx, single_text in enumerate(text):
                    token_ids = self.tokenizer.encode(single_text, max_length=self.max_length,
                                                      add_special_tokens=True)
                    sent_lens_list[text_idx].append(len(token_ids) - 1)
                    mask_ids = np.arange(self.max_length) <= len(token_ids)
                    pad_len = self.max_length - len(token_ids)
                    token_ids.extend([pad_id] * pad_len)
                    token_ids_list[text_idx].append(token_ids)
                    data_id_list[text_idx].append(item[src_data_id])
                    mask_ids_list[text_idx].append(mask_ids)
            else:
                token_ids = self.tokenizer.encode(text, max_length=self.max_length,
                                                  add_special_tokens=True)
                sent_lens_list.append(len(token_ids) - 1)
                mask_ids = np.arange(self.max_length) <= len(token_ids)
                pad_len = self.max_length - len(token_ids)
                token_ids.extend([pad_id] * pad_len)
                token_ids_list.append(token_ids)
                data_id_list.append(item[src_data_id])
                mask_ids_list.append(mask_ids)

        if prefix:
            token_field = '{}_tokens'.format(prefix)
            mask_field = '{}_masks'.format(prefix)
            data_id_field = '{}_data_ids'.format(prefix)
            sent_len_field = '{}_sent_lens'.format(prefix)
        else:
            token_field = 'tokens'
            mask_field = 'masks'
            data_id_field = 'data_ids'
            sent_len_field = 'sent_lens'
        if not is_text_list:
            batch = {token_field: np.array(token_ids_list, dtype=np.long),
                     sent_len_field: np.array(sent_lens_list, dtype=np.long),
                     mask_field: np.array(mask_ids_list),
                     data_id_field: data_id_list}
        else:
            batch = {token_field: [np.array(t, dtype=np.long) for t in token_ids_list],
                     sent_len_field: [np.array(t, dtype=np.long) for t in sent_lens_list],
                     mask_field: [np.array(t) for t in mask_ids_list],
                     data_id_field: data_id_list}
        return batch

    def __prefetch(self):
        for batch_idx in range(10):
            raw_batch = next(self._data)
            self.input_queue.put(raw_batch)
            self._batch_in_queue += 1

    def __iter__(self):
        for raw_batch in self._data:
            if self._batch_in_queue:
                batch = self.output_queue.get()
                yield self.batch2tensor(batch)
                self.input_queue.put(raw_batch)

        for _ in range(self._batch_in_queue):
            batch = self.output_queue.get()
            yield self.batch2tensor(batch)

    def batch2tensor(self, batch):
        new_batch = {}
        for key, val in batch.items():
            if 'data_ids' in key:
                new_batch[key] = val
            else:
                if isinstance(val, list) and isinstance(val[0], np.ndarray):
                    t = []
                    for single_np in val:
                        single_t = torch.tensor(single_np)
                        if torch.cuda.is_available():
                            single_t = single_t.cuda()
                        t.append(single_t)
                else:
                    t = torch.tensor(val)
                    if torch.cuda.is_available():
                        t = t.cuda()
                new_batch[key] = t

        return new_batch

    def __del__(self):
        self.input_queue.close()
        self.output_queue.close()
        for worker in self.workers:
            worker.terminate()
