# -*- coding: UTF-8 -*-
from munch import Munch
from wsdm_digg.constants import *
from wsdm_digg.reranking.dataloader import RerankDataLoader


def test_data_loader():
    args = Munch({
        'batch_size': 4,
        'max_len': 512
    })
    mode_name = 'bert-base-uncased'
    tokenizer = MODEL_DICT[mode_name]['tokenizer_class'].from_pretrained(mode_name)
    loader = RerankDataLoader(DATA_DIR + 'baseline_rerank.jsonl',
                              tokenizer,
                              args, 'eval')
    step = 0
    for _ in range(10):
        for batch in loader:
            if step == 0:
                print(batch)
            print(step)
            step += 1
