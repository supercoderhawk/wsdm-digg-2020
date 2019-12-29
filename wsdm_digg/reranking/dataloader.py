# -*- coding: UTF-8 -*-
import torch
from torch.multiprocessing import Queue, Process
import numpy as np
from pysenal import get_chunk, read_jsonline_lazy


class RerankDataLoader(object):
    def __init__(self, src_filename, tokenizer, args):
        self.src_filename = src_filename
        self.tokenizer = tokenizer
        self.batch_size = args.batch_size
        self.max_len = args.max_len

    def __iter__(self):
        return iter(RerankDataIterator(self))


class RerankDataIterator(object):
    def __init__(self, loader):
        self.loader = loader
        self.src_filename = loader.src_filename
        self.input_queue = Queue(-1)
        self.output_queue = Queue(-1)
        self.num_workers = 8
        self.batch_size = loader.batch_size
        self.tokenizer = loader.tokenizer
        self.max_len = loader.max_len
        self._batch_count_in_queue = 0
        self._data = get_chunk(read_jsonline_lazy(self.src_filename), self.batch_size)

        self.workers = []
        for _ in range(self.num_workers):
            worker = Process(target=self._data_loop)
            self.workers.append(worker)
        self.__prefetch()
        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def __iter__(self):
        for raw_batch in self._data:
            self.input_queue.put(raw_batch)
            yield self.batch2tensor(self.output_queue.get())
        if self._batch_count_in_queue:
            for _ in range(self._batch_count_in_queue):
                yield self.batch2tensor(self.output_queue.get())

    def batch2tensor(self, batch):
        new_batch = {}
        for key in batch:
            t = torch.tensor(batch[key])
            if torch.cuda.is_available():
                t = t.cuda()
            new_batch[key] = t
        return new_batch

    def __prefetch(self):
        for _ in range(10):
            self.input_queue.put(next(self._data))
            self._batch_count_in_queue += 1

    def _data_loop(self):
        while True:
            raw_batch = self.input_queue.get()
            true_part = self.encode_text(raw_batch, 'query', 'true_doc', 'true')
            false_part = self.encode_text(raw_batch, 'query', 'false_doc', 'false')
            batch = {**true_part, **false_part}
            self.output_queue.put(batch)

    def encode_text(self, raw_batch, query_field, doc_field, prefix):
        max_len = (self.max_len - 2) // 2
        pad_id = self.tokenizer.pad_token_id
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        token_ids_list = []
        segment_ids_list = []
        mask_ids_list = []
        for item in raw_batch:
            query_str = item[query_field]
            query_ids = self.tokenizer.encode(query_str,
                                              max_length=max_len,
                                              add_special_tokens=False)
            doc_str = item[doc_field]
            doc_ids = self.tokenizer.encode(doc_str,
                                            max_length=max_len,
                                            add_special_tokens=False)
            token_ids = [cls_id] + query_ids + [sep_id] + doc_ids + [sep_id]
            token_ids = token_ids[:self.max_len]
            sent_len = len(token_ids)
            token_ids = token_ids + [pad_id] * (self.max_len - len(token_ids))
            query_len = len(query_ids) + 1
            doc_len = self.max_len - len(query_ids) - 1
            segment_ids = [0] * query_len + [1] * doc_len
            mask_ids = np.arange(self.max_len) <= sent_len
            mask_ids_list.append(mask_ids)
            token_ids_list.append(token_ids)
            segment_ids_list.append(segment_ids)

        token_np = np.array(token_ids_list)
        segment_np = np.array(segment_ids_list)
        mask_np = np.array(mask_ids_list)
        batch = {'{}_token'.format(prefix): token_np,
                 '{}_segment'.format(prefix): segment_np,
                 '{}_mask'.format(prefix): mask_np}
        return batch

    def __del__(self):
        if self.workers:
            for worker in self.workers:
                worker.terminate()
            self.workers = []
