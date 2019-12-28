# -*- coding: UTF-8 -*-
from wsdm_digg.benchmark.evaluator import *
from wsdm_digg.constants import *

# print(Evaluator().evaluation_map(RESULT_DIR + 'baseline.jsonl'))
print(Evaluator().evaluation_recall(RESULT_DIR + 'baseline.jsonl', top_n=100))
