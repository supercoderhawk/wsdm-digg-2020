# -*- coding: UTF-8 -*-
import random
from pysenal import read_jsonline, write_jsonline
from wsdm_digg.constants import *


class DataSplitter(object):
    def __init__(self, src_filename):
        self.src_filename = src_filename

    def split(self):
        items = read_jsonline(self.src_filename)
        random.shuffle(items)
        random.shuffle(items)
        random.shuffle(items)
        training_count = int(len(items) * 0.9)
        training_items = items[:training_count]
        test_items = items[training_count:]
        write_jsonline(DATA_DIR + 'train.jsonl', training_items)
        write_jsonline(DATA_DIR + 'test.jsonl', test_items)
        # random.shuffle(test_items)
        # random.shuffle(test_items)
        # random.shuffle(test_items)
        # # only use 2000 items as rerank validation (named as test_rerank) data to save validation time
        # write_jsonline(DATA_DIR + 'test_rerank.jsonl', test_items[:2000])


if __name__ == '__main__':
    DataSplitter(DATA_DIR + 'train_release.jsonl').split()
