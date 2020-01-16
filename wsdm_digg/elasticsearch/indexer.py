# -*- coding: UTF-8 -*-
import json
import time
import traceback
import requests
from multiprocessing import Pool
from pysenal import read_json, read_jsonline_lazy, get_logger, get_chunk
from wsdm_digg.constants import *


class ElasticSearchIndexer(object):
    logger = get_logger('elastic indexer')
    parallel_size = 2
    retry_count = 10

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
        if ret.status_code != 200:
            raise Exception('setting es error, {}'.format(ret.text))

        ret = requests.put(mapping_url, data=field_data, headers=headers)
        if ret.status_code != 200:
            raise Exception('create index error, {}'.format(ret.text))
        self.logger.info('create index success')

    def delete_index(self):
        requests.delete(self.base_url)

    def indexing_runner(self):
        filename = CANDIDATE_FILENAME
        pool = Pool(self.parallel_size)
        start = time.time()
        count = 0
        failed_doc_list = []
        for item_chunk in get_chunk(read_jsonline_lazy(filename), 500):
            ret = pool.map(self.index_doc, item_chunk)
            failed_doc_list.extend([i for i in ret if i])
            duration = time.time() - start
            count += len(item_chunk)
            msg = '{} completed, {}min {:.2f}s'.format(count, duration // 60, duration % 60)
            self.logger.info(msg)

        for doc in failed_doc_list:
            self.index_doc(doc)

    def index_doc(self, doc):
        base_url = self.base_url + '/_doc/{}'
        headers = {"Content-Type": "application/json"}
        url = base_url.format(doc['paper_id'])
        if doc['keywords'] == '':
            doc['keywords'] = []
        else:
            doc['keywords'] = doc['keywords'].split(';')
        doc['TA'] = doc.get('title', '') + ' ' + doc.get('abstract', '')
        keyword_str = ' '.join(doc['keywords']).lower()
        if keyword_str:
            doc['TAK'] = doc['TA'] + ' ' + keyword_str
        else:
            doc['TAK'] = doc['TA']
        if not doc['paper_id'].strip():
            return

        input_str = json.dumps(doc)
        if not input_str:
            return

        try:
            ret = requests.put(url, input_str, headers=headers)
        except:
            count = 0
            is_success = False
            while count < self.retry_count:
                count += 1
                try:
                    ret = requests.put(url, input_str, headers=headers)
                except:
                    continue
                is_success = True
                break
            if not is_success:
                err_msg = '{} index failed, {}'.format(doc['paper_id'], traceback.format_exc())
                self.logger.error(err_msg)
                return doc
        if ret.status_code != 200 and ret.status_code != 201:
            msg = '{} error {} {}'.format(json.dumps(doc), ret.status_code, ret.text)
            self.logger.error(msg)


if __name__ == '__main__':
    ElasticSearchIndexer().run()
