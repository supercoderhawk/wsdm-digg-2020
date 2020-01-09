# -*- coding: UTF-8 -*-
import time
import pymagnitude
import hnswlib
import numpy as np
from pysenal import write_json, read_json
from wsdm_digg.constants import DATA_DIR


class VectorIndexer(object):
    dim = 768

    def __init__(self, src_filename, dest_filename):
        self.src_filename = src_filename
        self.dest_filename = dest_filename

    def read_paper_item(self):
        vectors = pymagnitude.Magnitude(self.src_filename)

        for paper_id, vec in vectors:
            yield paper_id, vec

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
        # print(list(self.idx2paper_id))
        paper_ids = [self.idx2paper_id[str(idx)] for idx in vec_id_list]
        return list(zip(paper_ids, distances))


if __name__ == '__main__':
    VectorIndexer(DATA_DIR + 'candidate_paper_scibert_vector.magnitude',
                  DATA_DIR + 'candidate_paper_scibert_index.bin').build_index()
