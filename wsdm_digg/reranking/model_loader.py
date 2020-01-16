# -*- coding: UTF-8 -*-
import torch
from wsdm_digg.reranking.plm_rerank import PlmRerank
from wsdm_digg.reranking.plm_knrm import PlmKnrm
from wsdm_digg.reranking.plm_conv_knrm import PlmConvKnrm
from wsdm_digg.reranking.plm_matchpyramid import PlmMatchPyramid
_MODEL_NAME_SET = {'plm', 'knrm', 'conv-knrm','mp','pairwise'}


def load_rerank_model(args):
    model_name = args.rerank_model_name
    if model_name not in _MODEL_NAME_SET:
        raise ValueError('model name {} is not implemented'.format(model_name))
    if model_name == 'plm' or model_name == 'pairwise':
        model = PlmRerank(args)
    elif model_name == 'knrm':
        model = PlmKnrm(args)
    elif model_name == 'conv-knrm':
        model = PlmConvKnrm(args)
    elif model_name == 'mp':
        model = PlmMatchPyramid(args)
    return model


def get_score_func(model, prefix=None, inference=False):
    def calculate(batch):
        if prefix:
            token_field = '{}_token'.format(prefix)
            segment_field = '{}_segment'.format(prefix)
            mask_field = '{}_mask'.format(prefix)
            query_lens_field = '{}_query_lens'.format(prefix)
            doc_lens_field = '{}_doc_lens'.format(prefix)
        else:
            token_field = 'token'
            segment_field = 'segment'
            mask_field = 'mask'
            query_lens_field = 'query_lens'
            doc_lens_field = 'doc_lens'
        if inference:
            with torch.no_grad():
                scores = model(token_ids=batch[token_field],
                               segment_ids=batch[segment_field],
                               token_mask=batch[mask_field],
                               query_lens=batch[query_lens_field],
                               doc_lens=batch[doc_lens_field])
                scores = scores.squeeze(1)
        else:
            scores = model(token_ids=batch[token_field],
                           segment_ids=batch[segment_field],
                           token_mask=batch[mask_field],
                           query_lens=batch[query_lens_field],
                           doc_lens=batch[doc_lens_field])
        return scores

    return calculate
