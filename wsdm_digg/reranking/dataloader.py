# -*- coding: UTF-8 -*-
import torch
import random
from itertools import chain
from torch.multiprocessing import Queue, Process
import numpy as np
from pysenal import get_chunk, read_jsonline_lazy, read_jsonline
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
        self.args = args

    def __iter__(self):
        return iter(RerankDataIterator(self))


class RerankDataIterator(object):
    def __init__(self, loader):
        self.loader = loader
        self.data_source = loader.data_source
        self.args = loader.args
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
            if self.args.lazy_loading:
                data = get_chunk(read_jsonline_lazy(self.data_source), self.batch_size)
            else:
                total_data = read_jsonline(self.data_source)
                random.shuffle(total_data)
                random.shuffle(total_data)
                random.shuffle(total_data)
                data = get_chunk(total_data, self.batch_size)
        elif isinstance(self.data_source, list):
            random.shuffle(self.data_source)
            random.shuffle(self.data_source)
            random.shuffle(self.data_source)
            data = get_chunk(self.data_source, self.batch_size)
        elif isinstance(self.data_source, dict):
            golden_filename = self.data_source['golden_filename']
            desc_id2item = {}
            for item in read_jsonline_lazy(golden_filename):
                desc_id2item[item['description_id']] = item
            search_filename = self.data_source['search_filename']
            topk = self.data_source['topk']
            searched_ids = set(self.data_source['searched_id_list'])

            def build_batch(search_item):
                qd_pairs = []
                desc_id = search_item['description_id']
                if desc_id in searched_ids:
                    return [[]]

                query_text = desc_id2item[desc_id][self.args.query_field]
                if self.args.rerank_model_name == 'pairwise':
                    docs = search_item['docs'][:topk]
                    for i, doc_id in enumerate(docs):
                        for p_doc_id in docs[:i] + docs[i+1:]:
                            raw_item = {'description_id': desc_id,
                                    'query': query_text, 'first_doc_id': doc_id,
                                    'second_doc_id':p_doc_id}
                            qd_pairs.append(raw_item)          
                else:    
                    for doc_id in search_item['docs'][:topk]:
                        raw_item = {'description_id': desc_id,
                                    'query': query_text, 'doc_id': doc_id}
                        qd_pairs.append(raw_item)

                return get_chunk(qd_pairs, self.batch_size)

            data = map(build_batch, read_jsonline_lazy(search_filename))
            data = chain.from_iterable(data)
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
        special_token_count = self.args.special_token_count
        if self.args.separate_query_doc:
            query_max_len = self.args.query_max_len
            doc_max_len = self.args.doc_max_len
        elif self.args.rerank_model_name =='pairwise':
            query_max_len = self.args.query_max_len
            doc_max_len = self.args.doc_max_len    
        else:    
            query_max_len = self.args.query_max_len
            doc_max_len = self.max_len - special_token_count - query_max_len
        
        pad_id = self.tokenizer.pad_token_id
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        token_ids_list = []
        segment_ids_list = []
        mask_ids_list = []
        query_len_list = []
        doc_len_list = []
        desc_id_set = set()

        for item in raw_batch:
            desc_id = item['description_id']
            desc_id_set.add(desc_id)
            query_str = item[query_field]
            query_ids = self.tokenizer.encode(query_str,
                                            max_length=query_max_len,
                                            add_special_tokens=False)
            if self.args.rerank_model_name == 'pairwise':
                query_ids = self.tokenizer.encode(query_str,
                                                max_length=query_max_len,
                                                add_special_tokens=False)
                if not prefix:                  
                    true_paper = get_paper(item['first_doc_id'])
                    false_paper = get_paper(item['second_doc_id'])                              
                    true_doc_str = true_paper['title'] + ' ' + true_paper['abstract']                    
                    false_doc_str = false_paper['title'] + ' ' + false_paper['abstract']                    
                else:
                    true_doc_str = item['true_doc']                    
                    false_doc_str = item['false_doc']                      
                true_doc_ids = self.tokenizer.encode(true_doc_str,
                                                max_length=doc_max_len,
                                                add_special_tokens=False)
                false_doc_ids = self.tokenizer.encode(false_doc_str,
                                                max_length=doc_max_len,
                                                add_special_tokens=False)                               
                if prefix == 'true':                                                                     
                    token_ids = [cls_id] + query_ids + [sep_id] + true_doc_ids + [sep_id] + false_doc_ids
                elif prefix == 'false':
                    token_ids = [cls_id] + query_ids + [sep_id] + false_doc_ids + [sep_id] + true_doc_ids
                elif not prefix:
                    token_ids = [cls_id] + query_ids + [sep_id] + true_doc_ids + [sep_id] + false_doc_ids    
                else:
                    raise ValueError('prefix error......')    
                token_ids = token_ids[:self.args.max_len]
                sent_len = len(token_ids)
                if sent_len < self.args.max_len:
                    token_ids.extend([pad_id]*(self.args.max_len - sent_len))
                segment_ids = [0] * self.args.max_len
                mask_ids = np.arange(self.max_len) <= sent_len
                mask_ids_list.append(mask_ids)
                token_ids_list.append(token_ids)
                segment_ids_list.append(segment_ids)
            else:
                if doc_field == 'doc_id':
                    doc_item = get_paper(item[doc_field])
                    doc_str = doc_item['title'] + ' ' + doc_item['abstract']
                else:
                    doc_str = item[doc_field]
                doc_ids = self.tokenizer.encode(doc_str,
                                                max_length=doc_max_len,
                                                add_special_tokens=False)

                query_ids = query_ids[:query_max_len]
                query_len_list.append(len(query_ids))
                # doc_len_list.append(len(doc_ids))
                doc_len_list.append(len(doc_ids))
                if self.args.separate_query_doc:
                    query_len = len(query_ids)
                    doc_len = len(query_ids)
                    query_ids.extend([pad_id] * (self.max_len - len(query_ids)))
                    doc_ids.extend([pad_id] * (self.max_len - len(doc_ids)))
                    segment_ids = [0] * self.max_len
                    query_mask_ids = np.arange(self.max_len) <= query_len
                    doc_mask_ids = np.arange(self.max_len) <= doc_len
                    if not token_ids_list:
                        token_ids_list =[[],[]]
                        segment_ids_list = [[],[]]
                        mask_ids_list = [[], []]
                        # query_len_list = [[], []]
                        # doc_len_list = [[], []]
                    token_ids_list[0].append(query_ids)
                    token_ids_list[1].append(doc_ids)
                    segment_ids_list[0].append(segment_ids)    
                    segment_ids_list[1].append(segment_ids)    
                    mask_ids_list[0].append(query_mask_ids)
                    mask_ids_list[1].append(doc_mask_ids)
                else:
                    if self.args.rerank_model_name != 'plm':
                        query_pad_len = query_max_len - len(query_ids)
                        query_ids.extend([pad_id] * query_pad_len)
                    doc_ids = doc_ids[:doc_max_len]

                    if special_token_count == 3:
                        token_ids = [cls_id] + query_ids + [sep_id] + doc_ids + [sep_id]
                    else:
                        token_ids = [cls_id] + query_ids + [sep_id] + doc_ids
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

        if self.args.separate_query_doc:
            token_np = [np.array(token_ids_list[0]),np.array(token_ids_list[1])]
            segment_np = [np.array(segment_ids_list[0]),np.array(segment_ids_list[1])]
            mask_np = [np.array(mask_ids_list[0]),np.array(mask_ids_list[1])]
        else:
            token_np = np.array(token_ids_list)
            segment_np = np.array(segment_ids_list)
            mask_np = np.array(mask_ids_list)    

        query_len_np = np.array(query_len_list)
        doc_len_np = np.array(doc_len_list)

        if self.mode in {'eval', 'inference'}:
            assert len(desc_id_set) == 1

        if prefix:
            token_field = '{}_token'.format(prefix)
            segment_field = '{}_segment'.format(prefix)
            mask_field = '{}_mask'.format(prefix)
            query_len_field = '{}_query_lens'.format(prefix)
            doc_len_field = '{}_doc_lens'.format(prefix)
        else:
            token_field = 'token'
            segment_field = 'segment'
            mask_field = 'mask'
            query_len_field = 'query_lens'
            doc_len_field = 'doc_lens'

        batch = {token_field: token_np,
                 segment_field: segment_np,
                 mask_field: mask_np,
                 query_len_field: query_len_np,
                 doc_len_field: doc_len_np}
        if self.mode in {'inference', 'eval'}:
            batch['raw'] = raw_batch
        return batch

    def __del__(self):
        if self.workers:
            for worker in self.workers:
                worker.terminate()
            self.workers = []
