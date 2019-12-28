# -*- coding: UTF-8 -*-
import json
import time
import requests
from multiprocessing import Pool
from pysenal import read_json, read_jsonline_lazy, get_logger, get_chunk
from wsdm_digg.constants import *


class ElasticSearchIndexer(object):
    logger = get_logger('elastic indexer')

    def __init__(self):
        self.base_url = ES_API_URL

    def run(self):
        self.delete_index()
        self.create_fields()
        self.indexing_runner()

    def create_fields(self):
        mapping_url = self.base_url + '/_mapping'
        headers = {"Content-Type": "application/json"}
        base_data = json.dumps(read_json(DATA_DIR + 'setting.json'))
        field_data = json.dumps(read_json(DATA_DIR + 'fields.json'))
        ret = requests.put(self.base_url, data=base_data, headers=headers)
        print(ret.status_code)
        print(ret.text)
        ret = requests.put(mapping_url, data=field_data, headers=headers)
        print(ret.status_code)
        print(ret.text)

    def delete_index(self):
        ret = requests.delete(self.base_url)
        print(ret.status_code, ret.text)

    def indexing_runner(self):
        filename = CANDIDATE_FILENAME
        pool = Pool(16)
        start = time.time()
        count = 0
        for item_chunk in get_chunk(read_jsonline_lazy(filename), 5000):
            pool.map(self.index_doc, item_chunk)
            duration = time.time() - start
            count += len(item_chunk)
            print('{} completed, {}min {}s'.format(count, duration // 60, duration % 60))

    def index_doc(self, doc):
        base_url = self.base_url + '/_doc/{}'
        headers = {"Content-Type": "application/json"}
        url = base_url.format(doc['paper_id'])
        if doc['keywords'] == '':
            doc['keywords'] = []
        else:
            doc['keywords'] = doc['keywords'].split(';')
        input_str = json.dumps(doc)
        if not input_str:
            return
        try:
            ret = requests.put(url, input_str, headers=headers)
        except:
            print(doc['paper_id'])
            return
        if ret.status_code != 200:
            if ret.status_code != 201:
                print(json.dumps(doc))
                print('error', ret.status_code, ret.text)


if __name__ == '__main__':
    ElasticSearchIndexer().run()
