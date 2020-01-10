# -*- coding: UTF-8 -*-
import time
import hnswlib
import numpy as np
from multiprocessing import Queue, Process
from pysenal import write_json, read_json, read_jsonline_lazy
from wsdm_digg.constants import DATA_DIR


class VectorIndexer(object):
    dim = 768

    def __init__(self, src_filename, dest_filename):
        self.src_filename = src_filename
        self.dest_filename = dest_filename
        self.input_queue = Queue(-1)
        self.output_queue = Queue(-1)
        self._data = read_jsonline_lazy(self.src_filename)
        self.worker_num = 8
        self.workers = []

        for _ in range(self.worker_num):
            worker = Process(target=self._worker_loop)
            self.workers.append(worker)
        self._count_in_queue = 0
        self.__prefetch()
        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def __prefetch(self):
        for _ in range(100):
            line = next(self._data)
            self.input_queue.put(line)
            self._count_in_queue += 1

    def _worker_loop(self):
        while True:
            line = self.input_queue.get()
            if line is None:
                break
            items = line.split()
            idx = items[0]
            vec = np.fromstring(' '.join(items[1:]), dtype=float, sep=' ')
            self.output_queue.put({'index': idx, 'vector': vec})

    def read_paper_item(self):
        for line in self._data:
            item = self.output_queue.get()
            yield item['index'], item['vector']
            self.input_queue.put(line)
        for _ in range(self._count_in_queue):
            item = self.output_queue.get()
            yield item['index'], item['vector']

    def build_index(self):
        indexer = hnswlib.Index(space='cosine', dim=self.dim)
        paper_id_list = []
        vector_list = []
        idx2paper_id = {}
        vector_id_list = []
        start = time.time()

        for vec_idx, (paper_id, vector) in enumerate(self.read_paper_item()):
            vector_id_list.append(vec_idx)
            paper_id_list.append(paper_id)
            vector_list.append(vector)
            idx2paper_id[vec_idx] = paper_id

        duration = time.time() - start
        msg_tmpl = 'vector loading completed time consumed {:.0f}min {:.2f}sec'
        print(msg_tmpl.format(duration // 60, duration % 60))
        num_elements = len(paper_id_list)
        indexer.init_index(max_elements=num_elements, ef_construction=200, M=16)
        # hnswlib only supports number based index,
        # therefore, mapper from number id to paper id is required to be saved
        indexer.add_items(vector_list, vector_id_list)
        indexer.set_ef(100)
        indexer.save_index(self.dest_filename)
        write_json(self.dest_filename + '.map', idx2paper_id)


class VectorSearcher(object):
    dim = 768

    def __init__(self, index_filename):
        self.indexer = hnswlib.Index(space='cosine', dim=self.dim)
        self.indexer.load_index(index_filename)
        self.idx2paper_id = read_json(index_filename + '.map')

    def query(self, vector, topk):
        vec_id_list, distances = self.indexer.knn_query([vector], k=topk)
        vec_id_list = vec_id_list[0]
        distances = distances[0]
        distances = 1.0 - distances
        paper_ids = [self.idx2paper_id[str(idx)] for idx in vec_id_list]
        return list(zip(paper_ids, distances))


if __name__ == '__main__':
    VectorIndexer(DATA_DIR + 'candidate_paper_scibert_vector.magnitude',
                  DATA_DIR + 'candidate_paper_scibert_index.bin').build_index()
