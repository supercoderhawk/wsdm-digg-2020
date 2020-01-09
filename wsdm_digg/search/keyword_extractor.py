# -*- coding: UTF-8 -*-
import re
import spacy
from textacy.keyterms import key_terms_from_semantic_network


class KeywordExtractor(object):
    nlp = spacy.load('en', disable=['ner', 'parser', 'textcat'])
    chunk_nlp = spacy.load('en', disable=['ner', 'textcat'])
    noun_chunk_stopwords = ('their ', 'the ', 'our ', 'my ', 'a ', 'an ', 'many ')
    es_special_char_regex = re.compile(r'(?P<PUNC>[+-=&|!(){}\[\]^"~*?:/])')

    def __init__(self):
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def textrank(self, input_data, topk, window_size, edge_weighting):
        if isinstance(input_data, spacy.tokens.Doc):
            doc = input_data
        elif isinstance(input_data, str):
            doc = self.nlp(input_data)
        else:
            raise TypeError('input is not spacy doc or text')
        textrank_ret = key_terms_from_semantic_network(doc=doc,
                                                       window_width=window_size,
                                                       edge_weighting=edge_weighting,
                                                       join_key_words=False,
                                                       n_keyterms=topk)
        result_terms = []
        for term, score in textrank_ret:
            if ' ' in term:
                result_terms.append('"' + self.format_term(term) + '"')
            else:
                result_terms.append(self.format_term(term))
        # result_terms = [term for term, score in textrank_ret]
        return result_terms

    def get_noun_chunk(self, text):
        noun_chunk_list = []
        for noun_chunk in self.chunk_nlp(text).noun_chunks:
            noun_chunk_list.append(noun_chunk.text)
        noun_chunk_list = self.noun_chunk_post_process(noun_chunk_list)
        return noun_chunk_list

    def noun_chunk_post_process(self, noun_chunk_list):
        processed_noun_chunks = []
        for noun_chunk in noun_chunk_list:
            if ' ' not in noun_chunk:
                continue
            if noun_chunk.startswith(self.noun_chunk_stopwords):
                new_noun_chunk = noun_chunk[noun_chunk.index(' ') + 1:].strip()
                if ' ' not in new_noun_chunk:
                    continue
                processed_noun_chunks.append(new_noun_chunk)
        return processed_noun_chunks

    def get_query_words(self, text, pos=False):
        pos_rules = {"NOUN", "PROPN", "ADJ"}
        query_words = []
        for token in self.nlp(text):
            if not token.is_stop and not token.is_punct:
                if pos:
                    if token.pos_ in pos_rules:
                        query_words.append(token.text)
                else:
                    query_words.append(token.text)
        return self.term_dedupe(query_words)

    def term_dedupe(self, terms):
        result_terms = []
        for term in terms:
            if term not in result_terms:
                result_terms.append(term)
        return result_terms

    def format_term(self, term):
        return self.es_special_char_regex.sub(r'\\\g<PUNC>', term)

    def format_terms(self, term_list):
        formatted_term_list = []
        for term in term_list:
            new_term = self.es_special_char_regex.sub(r'\\\g<PUNC>', term)
            formatted_term_list.append(new_term)
        return formatted_term_list
