# -*- coding: UTF-8 -*-
import os
import time
import argparse
from collections import OrderedDict
from munch import Munch
import torch
from pysenal import read_jsonline_lazy, read_json, append_jsonline
from wsdm_digg.reranking.dataloader import RerankDataLoader
from wsdm_digg.reranking.model_loader import load_rerank_model, get_score_func
from wsdm_digg.reranking.parse_args import parse_args
from wsdm_digg.utils import result_format
from wsdm_digg.constants import MODEL_DICT, DATA_DIR


class PlmRerankReranker(object):
    def __init__(self, model_info, batch_size, parser=None):
        self.batch_size = batch_size
        self.parser = parser
        if isinstance(model_info, str):
            self.model_path = model_info
            self.config = self.load_config()
            self.model = self.load_model(load_rerank_model(self.config), model_info)

            # for k in self.model.kernel.kernel_list:
            #     print(k.mean, k.stddev)

            model_dict = MODEL_DICT[self.config.plm_model_name]
            if 'path' in model_dict:
                tokenizer_path = model_dict['path'] + 'vocab.txt'
            else:
                tokenizer_path = self.config.model_name
            self.tokenizer = model_dict['tokenizer_class'].from_pretrained(tokenizer_path)
        elif isinstance(model_info, dict):
            self.model = model_info['model']
            self.config = model_info['config']
            self.tokenizer = model_info['tokenizer']
        else:
            raise ValueError('error')

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
        default_config = vars(parse_args(parser=self.parser))
        config_path = os.path.splitext(self.model_path)[0] + '.json'
        model_config = read_json(config_path)
        # config_dict = model_config
        config_dict = {**default_config, **model_config}
        config_dict['batch_size'] = self.batch_size
        config = Munch(config_dict)
        return config

    def get_searched_desc_id(self, filename):
        desc_ids = {item['description_id'] for item in read_jsonline_lazy(filename, default=[])}
        return desc_ids
    def rerank_pairwise_file(self, search_filename, golden_filename, dest_filename, topk, is_submit=False):
        self.model.eval()
        searched_desc_ids = self.get_searched_desc_id(dest_filename)
        data_source = {'search_filename': search_filename,
                       'golden_filename': golden_filename,
                       'topk': topk,
                       'searched_id_list': searched_desc_ids}
        loader = RerankDataLoader(data_source, self.tokenizer, self.config, 'eval')
        score_func = get_score_func(self.model, inference=True)

        start = time.time()
        desc_id2final = {}
        desc_id2id_score_list = {}
        desc_id2count = {}
        for batch_idx, batch in enumerate(loader):
            scores = score_func(batch)
            if torch.cuda.is_available():
                scores = scores.cpu()
            scores = scores.tolist()
            if isinstance(scores, float):
                scores = [scores]

            # desc_id = batch['raw'][0]['description_id']
            for i,s in zip(batch['raw'],scores):
                desc_id = i['description_id']
                fid = i['first_doc_id']
                if desc_id not in desc_id2id_score_list:
                    desc_id2id_score_list[desc_id] = {fid: [s]}
                else:
                    if fid not in desc_id2id_score_list[desc_id]:
                        desc_id2id_score_list[desc_id][fid] = [s]
                    else:    
                        desc_id2id_score_list[desc_id][fid].append(s)
                if len(desc_id2id_score_list[desc_id][fid]) == topk-1:
                    if desc_id not in desc_id2final:
                        desc_id2final[desc_id] = [(fid,sum(desc_id2id_score_list[desc_id][fid]))]
                    else:
                        desc_id2final[desc_id].append((fid,sum(desc_id2id_score_list[desc_id][fid])))
                    desc_id2id_score_list[desc_id].pop(fid)    
                    if len(desc_id2final[desc_id]) == topk: 
                        sorted_id_score_list = sorted(desc_id2final[desc_id], key=lambda i:i[1],reverse=True)
                        sorted_paper_ids = [idx for idx, _ in sorted_id_score_list]
                        result_item = {'description_id': desc_id, 'docs': sorted_paper_ids,
                               'docs_with_score': sorted_id_score_list}
                        append_jsonline(dest_filename, result_item) 
                        desc_id2final.pop(desc_id)      

        for desc_id,fid2score in desc_id2id_score_list.items():
            for fid in fid2score:
                if desc_id not in desc_id2final:
                    desc_id2final[desc_id] = [(fid,sum(fid2score[fid]))]
                else:
                    desc_id2final[desc_id].append((fid,sum(fid2score[fid])))

        for desc_id in desc_id2final:
            sorted_id_score_list = sorted(desc_id2final[desc_id], key=lambda i:i[1],reverse=True)
            sorted_paper_ids = [idx for idx, _ in sorted_id_score_list]
            result_item = {'description_id': desc_id, 'docs': sorted_paper_ids,
                    'docs_with_score': sorted_id_score_list}
            append_jsonline(dest_filename, result_item)
            # pass                
                

            # paper_id_score_list = list(zip([i['doc_id'] for i in batch['raw']], scores))
    def rerank_file(self, search_filename, golden_filename, dest_filename, topk, is_submit=False):
        assert topk >= self.batch_size
        assert topk % self.batch_size == 0

        self.model.eval()
        searched_desc_ids = self.get_searched_desc_id(dest_filename)
        data_source = {'search_filename': search_filename,
                       'golden_filename': golden_filename,
                       'topk': topk,
                       'searched_id_list': searched_desc_ids}
        loader = RerankDataLoader(data_source, self.tokenizer, self.config, 'eval')
        score_func = get_score_func(self.model, inference=True)

        start = time.time()

        desc_id2id_score_list = {}
        for batch_idx, batch in enumerate(loader):
            scores = score_func(batch)
            if torch.cuda.is_available():
                scores = scores.cpu()
            scores = scores.tolist()
            if isinstance(scores, float):
                scores = [scores]

            desc_id = batch['raw'][0]['description_id']
            paper_id_score_list = list(zip([i['doc_id'] for i in batch['raw']], scores))
            if desc_id not in desc_id2id_score_list:
                if len(paper_id_score_list) == topk:
                    sorted_id_score_list = sorted(paper_id_score_list,
                                                  key=lambda i: (i[1], i[0]), reverse=True)
                    sorted_paper_ids = [idx for idx, _ in sorted_id_score_list]
                    result_item = {'description_id': desc_id, 'docs': sorted_paper_ids,
                                   'docs_with_score': sorted_id_score_list}
                    append_jsonline(dest_filename, result_item)
                else:
                    desc_id2id_score_list[desc_id] = {'description_id': desc_id,
                                                      'docs': paper_id_score_list}
            else:
                desc_id2id_score_list[desc_id]['docs'].extend(paper_id_score_list)
                if len(desc_id2id_score_list[desc_id]['docs']) == topk:
                    topk_paper_id_score_list = desc_id2id_score_list[desc_id]['docs']
                    sorted_id_score_list = sorted(topk_paper_id_score_list,
                                                  key=lambda i: (i[1], i[0]), reverse=True)
                    sorted_paper_ids = [idx for idx, _ in sorted_id_score_list]
                    result_item = {'description_id': desc_id, 'docs': sorted_paper_ids,
                                   'docs_with_score': sorted_id_score_list}
                    append_jsonline(dest_filename, result_item)
                    desc_id2id_score_list.pop(desc_id)

        if desc_id2id_score_list:
            for desc_id, item in desc_id2id_score_list.items():
                topk_paper_id_score_list = item['docs']
                sorted_id_score_list = sorted(topk_paper_id_score_list,
                                              key=lambda i: (i[1], i[0]), reverse=True)
                sorted_paper_ids = [idx for idx, _ in sorted_id_score_list]
                result_item = {'description_id': desc_id, 'docs': sorted_paper_ids,
                               'docs_with_score': sorted_id_score_list}
                append_jsonline(dest_filename, result_item)

        duration = time.time() - start
        print('{}min {}sec'.format(duration // 60, duration % 60))
        if is_submit:
            result_format(dest_filename)

    def rerank(self, query, doc_id_list):
        self.model.eval()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-eval_search_filename', type=str, required=True)
    parser.add_argument('-golden_filename', type=str, required=True)
    parser.add_argument('-dest_filename', type=str, required=True)
    parser.add_argument('-model_path', type=str, required=True)
    parser.add_argument('-topk', type=int, default=20)
    parser.add_argument('-eval_batch_size', type=int, default=10)
    args = parser.parse_args()

    ranker = PlmRerankReranker(args.model_path, args.eval_batch_size, parser)
    ranker.rerank_file(args.eval_search_filename,
                       args.golden_filename,
                       args.dest_filename,
                       args.topk,
                       is_submit=True)


if __name__ == '__main__':
    main()
