# -*- coding: UTF-8 -*-
import os
import time
from collections import OrderedDict
from munch import Munch
import torch
from pysenal import read_jsonline_lazy, read_json, append_jsonline
from wsdm_digg.reranking.plm_rerank import PlmRerank
from wsdm_digg.reranking.dataloader import RerankDataLoader
from wsdm_digg.constants import MODEL_DICT, DATA_DIR


class PlmRerankReranker(object):
    def __init__(self, model_path, batch_size):
        self.model_path = model_path
        self.batch_size = batch_size
        self.config = self.load_config()
        self.model = self.load_model(PlmRerank(self.config), model_path)
        model_info = MODEL_DICT[self.config.model_name]
        if 'path' in model_info:
            tokenizer_path = model_info['path'] + 'vocab.txt'
        else:
            tokenizer_path = self.config.model_name
        self.tokenizer = model_info['tokenizer_class'].from_pretrained(tokenizer_path)

    def load_model(self, model, model_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = OrderedDict()
        # avoid error when load parallel trained model
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v
        model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def load_config(self):
        config_path = os.path.splitext(self.model_path)[0] + '.json'
        config_dict = read_json(config_path)
        config_dict['batch_size'] = self.batch_size
        config = Munch(config_dict)
        return config

    def get_searched_desc_id(self, filename):
        desc_ids = {item['description_id'] for item in read_jsonline_lazy(filename, default=[])}
        return desc_ids

    def rerank_file(self, search_filename, golden_filename, dest_filename, topk):
        searched_desc_ids = self.get_searched_desc_id(dest_filename)
        data_source = {'search_filename': search_filename,
                       'golden_filename': golden_filename,
                       'topk': topk,
                       'searched_id_list': searched_desc_ids}
        loader = RerankDataLoader(data_source, self.tokenizer, self.config, 'eval')
        start = time.time()
        for batch_idx, batch in enumerate(loader):
            scores = self.model(token_ids=batch['token'],
                                segment_ids=batch['segment'],
                                token_mask=batch['mask']).squeeze()
            if torch.cuda.is_available():
                scores = scores.cpu()
            scores = scores.tolist()
            desc_id = batch['raw'][0]['description_id']
            paper_ids_list = [i['doc_id'] for i in batch['raw']]
            id_with_score = sorted(zip(paper_ids_list, scores), key=lambda i: (i[1], i[0]), reverse=True)
            sorted_paper_ids = [idx for idx, _ in id_with_score]

            result_item = {'description_id': desc_id, 'docs': sorted_paper_ids,
                           'docs_with_score': id_with_score}
            append_jsonline(dest_filename, result_item)

        duration = time.time() - start
        print('{}min {}sec'.format(duration // 60, duration % 60))

    def rerank(self, query, doc_id_list):
        doc_len = len(doc_id_list)
        input_data = list(zip([query] * doc_len, doc_id_list))
        loader = RerankDataLoader(input_data, self.tokenizer, self.config, 'inference')
        id_with_score_list = []
        for batch in loader:
            scores = self.model(token_ids=batch['token'],
                                segment_ids=batch['segment'],
                                token_mask=batch['mask'])
            if torch.cuda.is_available():
                scores = scores.cpu()
            scores = scores.tolist()
            paper_ids_list = [i['doc_id'] for i in batch['raw']]
            id_with_score_list.append(list(zip(paper_ids_list, scores)))
        sorted_id_score_list = sorted(id_with_score_list, key=lambda i: (i[1], i[0]), reverse=True)
        sorted_paper_ids = [idx for idx, _ in sorted_id_score_list]
        return sorted_paper_ids


if __name__ == '__main__':
    topk = 100
    model_path = DATA_DIR + 'rerank/cite_textrank_top10_rerank_search_result' \
                            '/cite_textrank_top10_rerank_search_result_epoch_5_step_70000.model'
    search_filename = DATA_DIR + 'result/cite_textrank_top10.jsonl'
    golden_filename = DATA_DIR + 'test.jsonl'
    dest_filename = DATA_DIR + 'rerank_result/cite_textrank_top10_rerank_top{}.jsonl'.format(topk)
    PlmRerankReranker(model_path, topk).rerank_file(search_filename, golden_filename, dest_filename, topk)
