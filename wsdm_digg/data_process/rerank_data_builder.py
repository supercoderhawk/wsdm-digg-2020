# -*- coding: UTF-8 -*-
import os
import random
from multiprocessing import Pool
from pysenal import read_jsonline_lazy, get_chunk, append_jsonlines, index, read_lines
from wsdm_digg.elasticsearch.data import get_paper
from wsdm_digg.constants import DATA_DIR, RESULT_DIR


class RerankDataBuilder(object):
    def __init__(self, search_filename, golden_filename, dest_filename):
        self.search_filename = search_filename
        self.golden_filename = golden_filename
        self.dest_filename = dest_filename
        if os.path.exists(self.dest_filename):
            os.remove(self.dest_filename)
        self.candidate_paper_id_list = read_lines(DATA_DIR + 'candidate_paper_id.txt')

    def build_data(self, select_strategy='random'):
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
                                                    select_strategy)
                item.pop('docs')
                item.pop('keywords')
                new_item_chunk.append({**train_pair, **item, 'cites_text': cites_text})
            built_items = pool.map(self.build_single_query, new_item_chunk)
            append_jsonlines(self.dest_filename, built_items)

    def select_train_pair(self, doc_list, true_doc_id, select_strategy):
        offset = 50
        if select_strategy == 'search_result':
            true_idx = index(doc_list, true_doc_id, -1)
            if true_idx == -1 or true_idx + offset >= len(doc_list):
                false_paper_id = doc_list[-1]
            else:
                false_paper_id = doc_list[true_idx + offset]
        elif select_strategy == 'random':
            false_paper_id = random.choice(self.candidate_paper_id_list)
            if false_paper_id == true_doc_id:
                while True:
                    false_paper_id = random.choice(self.candidate_paper_id_list)
                    if false_paper_id != true_doc_id:
                        break
        else:
            raise ValueError('false instance select strategy error')
        return {'true_paper_id': true_doc_id, 'false_paper_id': false_paper_id}

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
    golden_filename = DATA_DIR + 'train.jsonl'
    # golden_filename = DATA_DIR + 'test.jsonl'
    search_filename = RESULT_DIR + 'cite_textrank_top10_train.jsonl'
    # search_filename = DATA_DIR + 'baseline_demo.jsonl'
    # dest_filename = DATA_DIR + 'train_rerank.jsonl'
    dest_filename = DATA_DIR + 'cite_textrank_top10_rerank_random.jsonl'
    builder = RerankDataBuilder(search_filename, golden_filename, dest_filename)
    builder.build_data(select_strategy='random')
