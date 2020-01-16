# -*- coding: UTF-8 -*-
import time
import argparse
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pysenal import *
from wsdm_digg.search.search import KeywordSearch
from wsdm_digg.constants import DATA_DIR, RESULT_DIR, SUBMIT_DIR
from wsdm_digg.benchmark.evaluator import Evaluator
from wsdm_digg.utils import result_format
from wsdm_digg.elasticsearch.data import get_paper

searcher = KeywordSearch()


class Benchmarker(object):
    def __init__(self,
                 dest_filename,
                 src_filename=DATA_DIR + 'test.jsonl',
                 batch_size=100,
                 parallel_count=20,
                 top_n=20,
                 is_submit=False,
                 is_final_submit=False):
        self.src_filename = src_filename
        if is_submit:
            self.src_filename = DATA_DIR + 'validation.jsonl'
            self.dest_filename = SUBMIT_DIR + dest_filename
        else:
            self.dest_filename = RESULT_DIR + dest_filename
        if is_final_submit:
            self.src_filename = DATA_DIR + 'test_release.jsonl'
            self.dest_filename = SUBMIT_DIR + dest_filename
        self.dest_csv_filename = os.path.splitext(self.dest_filename)[0] + '.csv'
        self.src_count = int(os.popen('wc -l {}'.format(self.src_filename)).read().split()[0])
        self.searched_id = self.get_searched_doc()
        self.batch_size = batch_size
        self.parallel_count = parallel_count
        self.top_n = top_n
        self.is_submit = is_submit
        self.is_final_submit = is_final_submit

    def batch_runner(self):
        start = time.time()
        pool = ThreadPool(self.parallel_count)

        while True:
            for doc_chunk in self.get_input_batch():
                ret = pool.map(self.single_query, doc_chunk)
                ret = [item for item in ret if item]
                append_jsonlines(self.dest_filename, ret)
            self.searched_id = self.get_searched_doc()
            duration = time.time() - start
            print('time consumed {}min {}sec'.format(duration // 60, duration % 60))
            if len(self.searched_id) == self.src_count:
                break

        duration = time.time() - start
        print('time consumed {}min {}sec'.format(duration // 60, duration % 60))
        if self.is_submit or self.is_final_submit:
            result_format(self.dest_filename, self.dest_csv_filename)
        if self.src_filename.endswith('test.jsonl'):
            eval_ret = Evaluator(self.src_filename).evaluation_map(self.dest_filename, top_n=3)
            print(eval_ret)

    def single_query(self, doc):
        try:
            # keywords = get_paper(doc['paper_id'])['keywords']
            # print(keywords)
            # ret = searcher.search(doc['description_text'], doc['cites_text'], self.top_n, keywords)
            ret = searcher.search(doc['description_text'], doc['cites_text'], self.top_n)
            return {'description_id': doc['description_id'], **ret}
        except Exception as e:
            # print(e)
            return None

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_filename', type=str, default=DATA_DIR + 'test.jsonl')
    parser.add_argument('-dest_filename', type=str, required=True, )
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-parallel_count', type=int, default=20)
    parser.add_argument('-top_n', type=int, default=20)
    parser.add_argument('-is_submit', action='store_true')
    args = parser.parse_args()

    Benchmarker(dest_filename=args.dest_filename,
                src_filename=args.src_filename,
                batch_size=args.batch_size,
                parallel_count=args.parallel_count,
                top_n=args.top_n,
                is_submit=args.is_submit
                ).batch_runner()


if __name__ == '__main__':
    main()
