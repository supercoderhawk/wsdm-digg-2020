# -*- coding: UTF-8 -*-
import time
from multiprocessing import Pool
from pysenal import *
from wsdm_digg.search.search import KeywordSearch
from wsdm_digg.constants import DATA_DIR

searcher = KeywordSearch()


class Benchmarker(object):
    def __init__(self,
                 dest_filename,
                 src_filename=DATA_DIR + 'test.jsonl',
                 batch_size=100,
                 parallel_count=20,
                 top_n=20):
        self.src_filename = src_filename
        self.dest_filename = dest_filename
        self.searched_id = self.get_searched_doc()
        self.batch_size = batch_size
        self.parallel_count = parallel_count
        self.top_n = top_n

    def batch_runner(self):
        start = time.time()
        pool = Pool(self.parallel_count)
        for doc_chunk in get_chunk(read_jsonline_lazy(self.src_filename), self.batch_size):
            ret = pool.map(self.single_query, doc_chunk)
            ret = [r for r in ret if r['docs']]
            append_jsonlines(self.dest_filename, ret)
        duration = time.time() - start
        print('time consumed {}min {}sec'.format(duration // 60, duration % 60))

    def single_query(self, doc):
        ret = searcher.search(doc['description_text'], self.top_n)
        return {'description_id': doc['description_id'], **ret}

    def get_input_batch(self):
        batch = []
        for doc in read_jsonline_lazy(self.src_filename):
            if doc['description_id'] in self.searched_id:
                continue
            batch.append(doc)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def get_searched_doc(self):
        searched_doc_id = []
        for doc in read_jsonline_lazy(self.dest_filename, default=[]):
            searched_doc_id.append(doc['description_id'])
        return searched_doc_id
