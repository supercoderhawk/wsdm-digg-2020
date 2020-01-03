# -*- coding: UTF-8 -*-
import os
import random
import argparse
from multiprocessing import Pool
from pysenal import read_jsonline_lazy, get_chunk, append_jsonlines, index, read_lines
from wsdm_digg.elasticsearch.data import get_paper
from wsdm_digg.constants import DATA_DIR, RESULT_DIR


class RerankDataBuilder(object):
    def __init__(self):
        self.args = self.parse_args()
        self.search_filename = self.args.search_filename
        self.golden_filename = self.args.golden_filename
        self.dest_filename = self.args.dest_filename
        if os.path.exists(self.dest_filename):
            os.remove(self.dest_filename)
        self.candidate_paper_id_list = read_lines(DATA_DIR + 'candidate_paper_id.txt')

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-search_filename', type=str, required=True)
        parser.add_argument('-golden_filename', type=str, required=True)
        parser.add_argument('-dest_filename', type=str, required=True)
        parser.add_argument('-select_strategy', type=str,
                            choices=['random', 'search_result_offset', 'search_result_false_top'],
                            required=True)
        parser.add_argument('-offset', type=int, default=50)
        args = parser.parse_args()
        return args

    def run(self):
        self.build_data()

    def build_data(self):
        pool = Pool(10)

        desc_id2item = {}

        for item in read_jsonline_lazy(self.golden_filename):
            desc_id = item['description_id']
            desc_id2item[desc_id] = item

        chunk_size = 500
        for item_chunk in get_chunk(read_jsonline_lazy(self.search_filename), chunk_size):
            new_item_chunk = []
            for item in item_chunk:
                true_paper_id = desc_id2item[item['description_id']]['paper_id']
                cites_text = desc_id2item[item['description_id']]['cites_text']
                train_pair = self.select_train_pair(item['docs'],
                                                    true_paper_id,
                                                    self.args.select_strategy)
                item.pop('docs')
                item.pop('keywords')
                new_item_chunk.append({**train_pair, **item, 'cites_text': cites_text})
            built_items = pool.map(self.build_single_query, new_item_chunk)
            append_jsonlines(self.dest_filename, built_items)

    def select_train_pair(self, doc_list, true_doc_id, select_strategy):
        offset = self.args.offset
        if select_strategy == 'search_result_offset':
            true_idx = index(doc_list, true_doc_id, -1)
            if true_idx == -1 or true_idx + offset >= len(doc_list):
                if len(doc_list) <= offset:
                    # when doc_list count is fewer than offset, used random selected false id
                    # because the result caused by following reasons will drop training result
                    # 1. the unusual description text will return 3 predefined paper id (in search.py)
                    # 2. too small topk in benchmark, and last instance of this result list has similar context of  true paper, will confused model
                    false_paper_id = self.random_choose_false_id(true_doc_id)
                else:
                    false_paper_id = doc_list[-1]
            else:
                false_paper_id = doc_list[true_idx + offset]
        elif select_strategy == 'random':
            false_paper_id = self.random_choose_false_id(true_doc_id)
        elif select_strategy == 'search_result_false_top':
            true_idx = index(doc_list, true_doc_id, -1)
            if true_idx == 0:
                false_paper_id = doc_list[1]
            else:
                false_paper_id = doc_list[0]
        else:
            raise ValueError('false instance select strategy error')
        return {'true_paper_id': true_doc_id, 'false_paper_id': false_paper_id}

    def random_choose_false_id(self, true_doc_id):
        false_paper_id = random.choice(self.candidate_paper_id_list)
        if false_paper_id == true_doc_id:
            while True:
                false_paper_id = random.choice(self.candidate_paper_id_list)
                if false_paper_id != true_doc_id:
                    break
        return false_paper_id

    def build_single_query(self, item):
        query = item['cites_text']
        true_paper = get_paper(item['true_paper_id'])
        false_paper = get_paper(item['false_paper_id'])
        true_text = true_paper['title'] + true_paper['abstract']
        false_text = false_paper['title'] + false_paper['abstract']
        train_item = {'query': query,
                      'true_doc': true_text,
                      'false_doc': false_text,
                      **item}
        train_item.pop('cites_text')
        return train_item


if __name__ == '__main__':
    RerankDataBuilder().run()
