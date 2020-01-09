# -*- coding: UTF-8 -*-
from pysenal import read_jsonline_lazy
from wsdm_digg.constants import *
from wsdm_digg.elasticsearch.data import get_paper


class DataViewer(object):
    def __init__(self):
        pass

    def view(self):
        for idx, item in enumerate(read_jsonline_lazy(DATA_DIR + 'test.jsonl')):
            if idx > 20:
                break
            paper = get_paper(item['paper_id'])
            print('============')
            print(item['description_text'])
            print('-------------')
            print(paper['title'])
            print(paper['abstract'])


if __name__ == '__main__':
    DataViewer().view()
