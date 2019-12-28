# -*- coding: UTF-8 -*-
from wsdm_digg.constants import *
from wsdm_digg.benchmark.benchmarker import Benchmarker

Benchmarker(RESULT_DIR + 'baseline.jsonl', top_n=100, parallel_count=10).batch_runner()
