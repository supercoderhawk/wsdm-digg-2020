# -*- coding: UTF-8 -*-
import torch
from torch.multiprocessing import Queue, Process
import numpy as np
from pysenal import get_chunk, read_jsonline_lazy
from wsdm_digg.elasticsearch.data import get_paper


class RerankDataLoader(object):
    def __init__(self, data_source, tokenizer, args, mode):
        if mode not in {'train', 'eval', 'inference'}:
            raise ValueError('data mode error')
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        self.mode = mode

    def __iter__(self):
        return iter(RerankDataIterator(self))


class RerankDataIterator(object):
    def __init__(self, loader):
        self.loader = loader
        self.data_source = loader.data_source
        self.num_workers = 8
        self.batch_size = loader.batch_size
        self.tokenizer = loader.tokenizer
        self.max_len = loader.max_len
        self.mode = loader.mode
        self._batch_count_in_queue = 0
        self._data = self.get_data()

        self.workers = []
        if self.mode in {'train', 'eval'}:
            self.input_queue = Queue(-1)
            self.output_queue = Queue(-1)
            for _ in range(self.num_workers):
                worker = Process(target=self._data_loop)
                self.workers.append(worker)
            self.__prefetch()
            for worker in self.workers:
                worker.daemon = True
                worker.start()

    def get_data(self):
        if isinstance(self.data_source, str):
            data = get_chunk(read_jsonline_lazy(self.data_source), self.batch_size)
        elif isinstance(self.data_source, list):
            data = get_chunk(self.data_source, self.batch_size)
        elif isinstance(self.data_source, dict):
            golden_filename = self.data_source['golden_filename']
            desc_id2item = {}
            for item in read_jsonline_lazy(golden_filename):
                desc_id2item[item['description_id']] = item
            search_filename = self.data_source['search_filename']
            topk = self.data_source['topk']
            searched_ids = self.data_source['searched_id_list']

            def build_batch(search_item):
                raw_batch = []
                desc_id = search_item['description_id']
                if desc_id in searched_ids:
                    return []

                query_text = desc_id2item[desc_id]['cites_text']
                for doc_id in search_item['docs'][:topk]:
                    raw_item = {'description_id': desc_id,
                                'query': query_text, 'doc_id': doc_id}
                    raw_batch.append(raw_item)
                return raw_batch

            data = map(build_batch, read_jsonline_lazy(search_filename))
        else:
            raise ValueError('data type error')
        return data

    def __iter__(self):
        if self.workers:
            for raw_batch in self._data:
                if not raw_batch:
                    continue
                self.input_queue.put(raw_batch)
                yield self.batch2tensor(self.output_queue.get())
            if self._batch_count_in_queue:
                for _ in range(self._batch_count_in_queue):
                    yield self.batch2tensor(self.output_queue.get())
        else:
            for raw_batch in self._data:
                if not raw_batch:
                    continue
                yield self.batch2tensor(self.encode_text(raw_batch, 'query', 'doc', None))

    def batch2tensor(self, batch):
        new_batch = {}
        for key in batch:
            if key in {'raw'}:
                new_batch[key] = batch[key]
                continue
            t = torch.tensor(batch[key])
            if torch.cuda.is_available():
                t = t.cuda()
            new_batch[key] = t
        return new_batch

    def __prefetch(self):
        prefetch_step = 10
        step = 0

        while True:
            raw_batch = next(self._data, None)
            if raw_batch is None:
                break
            if not raw_batch:
                continue
            self.input_queue.put(raw_batch)
            self._batch_count_in_queue += 1
            step += 1
            if step >= prefetch_step:
                break

    def _data_loop(self):
        while True:
            raw_batch = self.input_queue.get()
            if self.mode == 'train':
                true_part = self.encode_text(raw_batch, 'query', 'true_doc', 'true')
                false_part = self.encode_text(raw_batch, 'query', 'false_doc', 'false')
                batch = {**true_part, **false_part}
            elif self.mode == 'eval':
                batch = self.encode_text(raw_batch, 'query', 'doc_id', None)
            elif self.mode == 'inference':
                batch = self.encode_text(raw_batch, 'query', 'doc', None)
            self.output_queue.put(batch)

    def encode_text(self, raw_batch, query_field, doc_field, prefix):
        query_max_len = 100
        doc_max_len = self.max_len - 2 - query_max_len
        pad_id = self.tokenizer.pad_token_id
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        token_ids_list = []
        segment_ids_list = []
        mask_ids_list = []
        for item in raw_batch:
            query_str = item[query_field]
            query_ids = self.tokenizer.encode(query_str,
                                              max_length=query_max_len,
                                              add_special_tokens=False)
            if doc_field == 'doc_id':
                doc_item = get_paper(item[doc_field])
                doc_str = doc_item['title'] + ' ' + doc_item['abstract']
            else:
                doc_str = item[doc_field]
            doc_ids = self.tokenizer.encode(doc_str,
                                            max_length=doc_max_len,
                                            add_special_tokens=False)

            query_ids = query_ids[:query_max_len]
            doc_ids = doc_ids[:doc_max_len]
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
        if prefix:
            token_field = '{}_token'.format(prefix)
            segment_field = '{}_segment'.format(prefix)
            mask_field = '{}_mask'.format(prefix)
        else:
            token_field = 'token'
            segment_field = 'segment'
            mask_field = 'mask'
        batch = {token_field: token_np,
                 segment_field: segment_np,
                 mask_field: mask_np}
        if self.mode in {'inference', 'eval'}:
            batch['raw'] = raw_batch
        return batch

    def __del__(self):
        if self.workers:
            for worker in self.workers:
                worker.terminate()
            self.workers = []
