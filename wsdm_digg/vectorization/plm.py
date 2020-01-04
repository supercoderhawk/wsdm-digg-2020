# -*- coding: UTF-8 -*-
import argparse
import h5py
import numpy as np
import torch
from pysenal import read_jsonline_lazy, get_chunk
from torch.multiprocessing import Queue, Process
from wsdm_digg.constants import MODEL_DICT


class VectorizationDataLoader(object):
    def __init__(self, src_filename, tokenizer, batch_size, max_length):
        self.src_filename = src_filename
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_queue = Queue(-1)
        self.output_queue = Queue(-1)
        self._data = get_chunk(read_jsonline_lazy(self.src_filename), batch_size)
        self.worker_num = 8
        self.workers = []
        self._batch_in_queue = 0

        for _ in range(self.worker_num):
            worker = Process(target=self._worker_loop)
            self.workers.append(worker)

        self.__prefetch()
        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def _worker_loop(self):
        pad_id = self.tokenizer.pad_token_id
        while True:
            raw_batch = self.input_queue.get()
            if raw_batch is None:
                break
            token_ids_list = []
            paper_id_list = []
            mask_ids_list = []
            for item in raw_batch:
                text = item['title'] + ' ' + item['abstract']
                token_ids = self.tokenizer.encode(text, max_length=self.max_length,
                                                  add_special_tokens=True)
                mask_ids = np.arange(self.max_length) <= len(token_ids)
                pad_len = self.max_length - len(token_ids)
                token_ids.extend([pad_id] * pad_len)
                token_ids_list.append(token_ids)
                paper_id_list.append(item['paper_id'])
                mask_ids_list.append(mask_ids)

            batch = {'tokens': np.array(token_ids_list, dtype=np.long),
                     'masks': np.array(mask_ids_list),
                     'paper_ids': paper_id_list}
            self.output_queue.put(batch)

    def __prefetch(self):
        for _ in range(10):
            raw_batch = next(self._data)
            self.input_queue.put(raw_batch)
            self._batch_in_queue += 1

    def __iter__(self):
        for raw_batch in self._data:
            if self._batch_in_queue:
                batch = self.output_queue.get()
                batch['tokens'] = torch.tensor(batch['tokens'])
                batch['masks'] = torch.tensor(batch['masks'])
                if torch.cuda.is_available():
                    batch['tokens'] = batch['tokens'].cuda()
                    batch['masks'] = batch['masks'].cuda()
                yield batch
                self.input_queue.put(raw_batch)

        for _ in range(self._batch_in_queue):
            batch = self.output_queue.get()
            batch['tokens'] = torch.tensor(batch['tokens'])
            batch['masks'] = torch.tensor(batch['masks'])
            if torch.cuda.is_available():
                batch['tokens'] = batch['tokens'].cuda()
                batch['masks'] = batch['masks'].cuda()
            yield batch

    def __del__(self):
        self.input_queue.close()
        self.output_queue.close()
        for worker in self.workers:
            worker.terminate()


class PlmVectorization(object):
    def __init__(self):
        self.args = self.parse_args()
        plm_model_name = self.args.plm_model_name
        if plm_model_name not in MODEL_DICT:
            raise ValueError('model name is not supported.')
        model_info = MODEL_DICT[plm_model_name]
        if 'path' in model_info:
            plm_model_name = model_info['path']
        self.plm_model = model_info['model_class'].from_pretrained(plm_model_name)

        if torch.cuda.is_available():
            self.plm_model.cuda()
        if 'path' in model_info:
            tokenizer_path = model_info['path'] + 'vocab.txt'
        else:
            tokenizer_path = self.args.plm_model_name

        self.tokenizer = model_info['tokenizer_class'].from_pretrained(tokenizer_path)
        self.loader = VectorizationDataLoader(self.args.src_filename, self.tokenizer, self.args.batch_size,
                                              self.args.max_length)

    def run(self):
        f = h5py.File(self.args.dest_filename, 'a')
        for batch in self.loader:
            output = self.plm_model(input_ids=batch['tokens'],
                                    attention_mask=batch['masks'])[0]
            for sent_embed, paper_id in zip(output[:, 0], batch['paper_ids']):
                if torch.cuda.is_available():
                    sent_embed = sent_embed.cpu()
                sent_embed = sent_embed.detach().numpy()
                f.create_dataset('{}'.format(paper_id), data=sent_embed)
        f.close()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-plm_model_name', type=str, required=True, help='')
        parser.add_argument('-src_filename', type=str, required=True, help='')
        parser.add_argument('-dest_filename', type=str, required=True, help='')
        parser.add_argument('-batch_size', type=int, default=3, help='')
        parser.add_argument('-max_length', type=int, default=512, help='')
        args = parser.parse_args()
        return args


if __name__ == '__main__':
    PlmVectorization().run()
