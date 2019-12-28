# -*- coding: UTF-8 -*-
import os
import spacy
import numpy as np
import pandas as pd
from pysenal.io import *
from wsdm_digg.constants import *


class RawDataFormatter(object):
    nlp = spacy.load('en', disable=['ner', 'parser', 'textcat'])

    def __init__(self, dirname):
        self.dirname = dirname
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def run(self):
        self.candidate_formatter()
        self.train_data_formatter()
        self.validation_data_formatter()

    def candidate_formatter(self):
        candidate_filename = self.dirname + 'candidate_paper_for_wsdm2020.csv'
        dest_filename = os.path.splitext(candidate_filename)[0] + '.jsonl'
        if os.path.exists(dest_filename):
            os.remove(dest_filename)
        df = pd.read_csv(candidate_filename).replace(np.nan, '', regex=True)
        for idx, row in df.iterrows():
            item = {'paper_id': row['paper_id'], 'title': row['title'],
                    'abstract': row['abstract'], 'journal': row['journal'],
                    'keywords': row['keywords']}
            append_jsonline(dest_filename, item)

    def train_data_formatter(self):
        train_filename = self.dirname + 'train_release.csv'
        dest_filename = os.path.splitext(train_filename)[0] + '.jsonl'
        desc_id_set = set()
        if os.path.exists(dest_filename):
            os.remove(dest_filename)
        df = pd.read_csv(train_filename).replace(np.nan, '', regex=True)
        for idx, row in df.iterrows():
            cites_text = self.extract_cites_sent(row['description_text'])
            desc_id = row['description_id']
            if desc_id in desc_id_set:
                continue
            desc_id_set.add(desc_id)
            item = {'description_id': desc_id,
                    'paper_id': row['paper_id'],
                    'description_text': row['description_text'],
                    'cites_text': cites_text}

            append_jsonline(dest_filename, item)

    def validation_data_formatter(self):
        valid_filename = self.dirname + 'validation.csv'
        dest_filename = os.path.splitext(valid_filename)[0] + '.jsonl'
        if os.path.exists(dest_filename):
            os.remove(dest_filename)
        df = pd.read_csv(valid_filename).replace(np.nan, '', regex=True)
        for idx, row in df.iterrows():
            cites_text = self.extract_cites_sent(row['description_text'])
            item = {'description_id': row['description_id'],
                    'description_text': row['description_text'],
                    'cites_text': cites_text}
            append_jsonline(dest_filename, item)

    def extract_cites_sent(self, text):
        doc = self.nlp(text)
        id_str = '[[**##**]]'
        sent_text = ''
        for sent in doc.sents:
            if id_str in sent.text:
                sent_text += ' ' + sent.text
        return sent_text if sent_text else text


if __name__ == '__main__':
    RawDataFormatter(DATA_DIR).run()
