# -*- coding: UTF-8 -*-
from pysenal.io import *
from pysenal.utils import *
from wsdm_digg.constants import DATA_DIR


class Evaluator(object):
    def __init__(self, true_filename=DATA_DIR + 'test.jsonl'):
        self.true_filename = true_filename
        self.true_data = self.true_data_loader()

    def evaluation_recall(self, pred_filename, top_n):
        hit_count = 0
        total_count = 0
        for doc in read_jsonline_lazy(pred_filename):
            true_paper_id = self.true_data[doc['description_id']]
            val = index(doc['docs'], true_paper_id, top_n)
            if val < top_n:
                hit_count += 1
            total_count += 1
        recall = hit_count / total_count
        return recall

    def evaluation_map(self, pred_filename, top_n=3):
        sum_ap = 0
        query_count = 0
        for doc in read_jsonline_lazy(pred_filename):
            true_paper_id = self.true_data[doc['description_id']]
            val = index(doc['docs'], true_paper_id, top_n)
            if val < top_n:
                sum_ap += 1 / (val + 1)
            query_count += 1
        map = sum_ap / query_count
        return map

    def true_data_loader(self):
        true_data = {}
        for doc in read_jsonline_lazy(self.true_filename):
            true_data[doc['description_id']] = doc['paper_id']
        return true_data
