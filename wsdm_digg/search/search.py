# -*- coding: UTF-8 -*-
import re
import requests
from .keyword_extractor import KeywordExtractor
from ..constants import ES_API_URL


class KeywordSearch(object):
    field_weight = {'title': 2,
                    'abstract': 3,
                    'TA': 10,
                    'keywords': 1}
    es_special_char_regex = re.compile(r'(?P<PUNC>[\\+-=&|!(){}\[\]^"~*?:/])')
    cites_person_name_regex = re.compile(r'(?:(?:[A-Z][a-z]{1,20} ?){1,3}(?:, )?){1,5}et al')
    default_result = ['55a38fe7c91b587b095b0d1c',
                      '55a4eb3e65ceb7cb02dbff7c',
                      '55a3a74065ce5cd7b3b2db98']

    def __init__(self):
        self.extractor = KeywordExtractor()
        self.nlp = self.extractor.nlp
        self.search_url = ES_API_URL + '/_search'
        self.headers = {"Content-Type": "application/json"}

    def search(self, text, cites_text, top_n, paper_keywords=None):
        if not text:
            raise ValueError('input search text is empty')

        noun_chunks = self.extractor.get_noun_chunk(cites_text)
        noun_chunks = self.format_terms(noun_chunks)
        noun_chunks = ['"' + noun + '"' for noun in noun_chunks]
        query_words = self.extractor.get_query_words(cites_text)
        query_words = self.format_terms(query_words)
        query_terms = noun_chunks + query_words
        if not query_terms:
            query_terms = self.extractor.get_query_words(text)
            query_terms = self.format_terms(query_terms)

        cites_keywords = self.extractor.textrank(cites_text, 15, window_size=2,
                                                 edge_weighting='binary')
        query_terms = query_terms + cites_keywords

        important_keywords = self.format_terms(self.extractor.get_query_words(text))
        query_terms = query_terms + important_keywords

        query_terms = [term for term in query_terms if term.strip()]
        if not query_terms:
            query_terms = self.format_terms([text])

        # paper_keywords = self.format_terms(paper_keywords)
        # paper_keywords = ['"' + k + '"' if ' ' in k else k for k in paper_keywords]
        query_dict = {'TA': query_terms}
        es_query_obj = self.build_es_query_string_object(query_dict, top_n)
        ret = requests.post(self.search_url, json=es_query_obj, headers=self.headers)
        searched_paper_id = []
        if ret.status_code == 200:
            paper_list = ret.json()['hits']['hits']
            # searched_paper_id = self.prf(query_terms, paper_list[:1], top_n)
            for doc in paper_list:
                searched_paper_id.append(doc['_id'])
            # When query text is abnormal , the ES search result will be less than three
            # then replace it with default value just to avoid submit error
            if len(searched_paper_id) < 3:
                searched_paper_id = self.default_result
        else:
            print('search error', ret.text)
        return {'docs': searched_paper_id, 'keywords': query_terms}

    num_regex = re.compile(r'((\d+(.\d+)?-)+(\d+(.\d+)?)$)|(\d+(.\d+)?)')
    num_char = '<num>'

    def prf(self, query_terms, paper_list, top_n):
        text = ' '.join([p['_source']['TA'] for p in paper_list])
        keywords = self.extractor.textrank(text, 5, window_size=10,
                                           edge_weighting='coor_freq')

        query_dict = {'TA': query_terms + self.format_terms(keywords)}
        es_query_obj = self.build_es_query_string_object(query_dict, top_n)
        ret = requests.post(self.search_url, json=es_query_obj, headers=self.headers)
        searched_paper_id = []
        if ret.status_code == 200:
            for doc in ret.json()['hits']['hits']:
                searched_paper_id.append(doc['_id'])
            # When query text is unusual , the ES search result will be less than three
            # then replace it with default value just to avoid submit error
            if len(searched_paper_id) < 3:
                searched_paper_id = self.default_result
        else:
            print('search error', ret.text)
        return searched_paper_id

    def build_es_query_string_object(self, query_dict, rows):
        query_segments = []
        for field, terms in query_dict.items():
            if not terms:
                continue
            segment = '{}:({})^{}'.format(field, ' OR '.join(terms), self.field_weight[field])
            query_segments.append(segment)
        query_str = ' OR '.join(query_segments)
        es_obj = {'query': {'query_string': {'query': query_str}},
                  'size': rows}
        return es_obj

    def format_terms(self, term_list):
        formatted_term_list = []
        for term in term_list:
            new_term = self.es_special_char_regex.sub(r'\\\g<PUNC>', term)
            formatted_term_list.append(new_term)
        return formatted_term_list

    def boost_terms(self, term_list, boost_val):
        boosted_terms = []
        for term in term_list:
            boosted_terms.append(term + '^{}'.format(boost_val))
        return boosted_terms

    def clean_name(self, term_list, name):
        cleaned_term_list = []
        for term in term_list:
            if term not in name:
                cleaned_term_list.append(term)
        return cleaned_term_list

    def extract_person_name(self, text):
        return ' '.join(self.cites_person_name_regex.findall(text))
