# -*- coding: UTF-8 -*-
import os
import spacy
import numpy as np
import pandas as pd
from pysenal.io import *
from wsdm_digg.constants import *


class RawDataFormatter(object):
    """
    1. transform raw candidate paper data, train validation and test data into json line format
    2. extract the citation sentence which has citation indicator  '[**##**]'
    """
    nlp = spacy.load('en', disable=['ner', 'parser', 'textcat'])

    def __init__(self, dirname):
        self.dirname = dirname
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def run(self):
        self.candidate_formatter()
        self.train_data_formatter()
        self.validation_data_formatter(self.dirname + 'validation.csv')
        self.validation_data_formatter(self.dirname + 'test_release.csv')

    def candidate_formatter(self):
        candidiate_paper_id_list = []
        candidate_filename = self.dirname + 'candidate_paper_for_wsdm2020.csv'
        dest_filename = os.path.splitext(candidate_filename)[0] + '.jsonl'
        if os.path.exists(dest_filename):
            os.remove(dest_filename)
        df = pd.read_csv(candidate_filename).replace(np.nan, '', regex=True)
        for idx, row in df.iterrows():
            item = {'paper_id': row['paper_id'], 'title': row['title'],
                    'abstract': row['abstract'], 'journal': row['journal'],
                    'keywords': row['keywords']}
            candidiate_paper_id_list.append(row['paper_id'])
            append_jsonline(dest_filename, item)
        write_lines(self.dirname + 'candidate_paper_id.txt', candidiate_paper_id_list)

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
            if not row['description_text'].strip():
                continue
            if desc_id in desc_id_set:
                continue
            desc_id_set.add(desc_id)
            item = {'description_id': desc_id,
                    'paper_id': row['paper_id'],
                    'description_text': row['description_text'],
                    'cites_text': cites_text}

            append_jsonline(dest_filename, item)

    def validation_data_formatter(self, src_filename, dest_filename=None):
        valid_filename = src_filename
        if dest_filename is None:
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
        id_str = '[**##**]'
        cite_text = ''
        for sent in doc.sents:
            sent_text = sent.text
            if id_str in sent_text:
                cite_text += ' ' + sent_text
        cite_text = cite_text.strip()
        return cite_text if cite_text else text


if __name__ == '__main__':
    RawDataFormatter(DATA_DIR).run()
