# -*- coding: UTF-8 -*-
import numpy as np
from pysenal import read_jsonline_lazy, append_jsonline
from wsdm_digg.vectorization.indexer import VectorIndexer
from wsdm_digg.constants import DATA_DIR, RESULT_DIR


class VectorReranker(object):
    def __init__(self, index_filename):
        self.indexer = VectorIndexer(index_filename)
        self.paper_id2vector = self.indexer.load_vector()

    def rerank(self, src_filename, dest_filename, vector_filename, topk=100):
        desc_id2vector = VectorIndexer(vector_filename).load_vector()
        for item in read_jsonline_lazy(src_filename):
            desc_id = item['description_id']
            paper_id_list = item['docs'][:topk]
            if desc_id in desc_id2vector:
                desc_vector = desc_id2vector[desc_id]
                paper_id_with_scores = []
                for paper_id in paper_id_list:
                    if paper_id not in self.paper_id2vector:
                        score = 0
                    else:
                        paper_vector = self.paper_id2vector[paper_id]
                        dot_score = np.dot(paper_vector, desc_vector)
                        norm_score = np.linalg.norm(paper_vector) * np.linalg.norm(desc_vector)
                        score = dot_score / norm_score
                    paper_id_with_scores.append((paper_id, score))
                sorted_id_scores = sorted(paper_id_with_scores, key=lambda i: (i[1], i[0]),
                                          reverse=True)
                reranked_paper_id_list = [p for p, s in sorted_id_scores]
            else:
                reranked_paper_id_list = paper_id_list
            result_item = {'description_id': desc_id, 'docs': reranked_paper_id_list}
            append_jsonline(dest_filename, result_item)


if __name__ == '__main__':
    reranker = VectorReranker(DATA_DIR + 'candidate_paper_dssm_loss_vector.txt')
    reranker.rerank(RESULT_DIR + 'only_TA.jsonl', RESULT_DIR + 'only_TA_vector_rerank_top200.jsonl',
                    DATA_DIR + 'test_dssm_loss_vector.txt', topk=200)
