# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from wsdm_digg.constants import MODEL_DICT


class PlmKnrm(nn.Module):
    model_name = 'knrm'

    def __init__(self, args):
        super().__init__()
        self.args = args
        plm_model_name = self.args.plm_model_name
        if plm_model_name not in MODEL_DICT:
            raise ValueError('model name is not supported.')
        model_info = MODEL_DICT[plm_model_name]
        if 'path' in model_info:
            plm_model_name = model_info['path']
        self.model = model_info['model_class'].from_pretrained(plm_model_name)
        if torch.cuda.is_available():
            self.model.cuda()
        self.mean_list = self.args.mean_list
        self.stddev_list = self.args.stddev_list
        assert len(self.mean_list) == len(self.stddev_list)
        self.query_max_len = self.args.query_max_len
        self.doc_max_len = self.args.max_len - self.query_max_len - self.args.special_token_count
        self.use_context_vector = self.args.use_context_vector
        self.context_merge_method = self.args.context_merge_method
        if self.use_context_vector:
            if self.context_merge_method == 'vector_concat':
                self.score_dim = len(self.mean_list) + self.args.dim_size
            elif self.context_merge_method == 'score_add':
                self.score_dim = len(self.mean_list)
                self.context_score_proj = nn.Linear(self.args.dim_size, 1, bias=True)
            else:
                raise ValueError('error...')
        else:
            self.score_dim = len(self.mean_list)
        self.score_proj = nn.Linear(self.score_dim, 1, bias=True)

    def forward(self, token_ids, segment_ids, token_mask, query_lens, doc_lens):
        contextualized_embed = self.model(input_ids=token_ids,
                                          attention_mask=token_mask,
                                          token_type_ids=segment_ids)[0]
        query_embed = contextualized_embed[:, 1:self.query_max_len + 1]
        doc_start_idx = self.query_max_len + 1
        doc_end_idx = doc_start_idx + self.doc_max_len
        doc_embed = contextualized_embed[:, doc_start_idx:doc_end_idx]

        sim_matrix = self.get_similarity_matrix(query_embed, doc_embed, query_lens, doc_lens)
        rbf_feature = self.get_kernel_pooling(sim_matrix)
        score = self.get_ranking_score(rbf_feature, contextualized_embed[:, 0])
        return score

    def get_similarity_matrix(self, query_embed, doc_embed, query_lens, doc_lens):
        batch_size = query_lens.size(0)
        query_sum = torch.sqrt(torch.sum(query_embed ** 2, dim=2).unsqueeze(2))
        doc_sum = torch.sqrt(torch.sum(doc_embed ** 2, dim=2).unsqueeze(1))

        sim_matrix = torch.bmm(query_embed, doc_embed.permute(0, 2, 1)) / torch.bmm(query_sum, doc_sum)
        # mask position with pad char in query and doc to value 0
        q_max_len = self.query_max_len
        d_max_len = self.doc_max_len
        q_range = torch.arange(q_max_len).unsqueeze(0).repeat(batch_size, 1)
        if torch.cuda.is_available():
            q_range = q_range.cuda()
        q_mask = q_range <= query_lens.unsqueeze(1)
        q_mask = q_mask.unsqueeze(2).repeat(1, 1, d_max_len)
        d_range = torch.arange(d_max_len).unsqueeze(0).repeat(batch_size, 1)
        if torch.cuda.is_available():
            d_range = d_range.cuda()
        d_mask = d_range <= doc_lens.unsqueeze(1)
        d_mask = d_mask.unsqueeze(1).repeat(1, q_max_len, 1)
        total_mask = ~(q_mask * d_mask)
        sim_matrix.masked_fill_(total_mask, 0.0)
        return sim_matrix

    def get_kernel_pooling(self, similarity_matrix):
        rbf_feature = None
        for mean, stddev in zip(self.mean_list, self.stddev_list):
            rbf_matrix = torch.exp(-(similarity_matrix - mean) ** 2 / (2 * stddev ** 2))
            if rbf_feature is None:
                rbf_feature = torch.sum(rbf_matrix, dim=2).unsqueeze(1)
            else:
                rbf_feature = torch.cat((rbf_feature, torch.sum(rbf_matrix, dim=2).unsqueeze(1)), dim=1)
        rbf_feature = torch.sum(torch.log(rbf_feature), dim=2)
        return rbf_feature

    def get_ranking_score(self, rbf_feature, context_feature):
        # score = torch.tanh(self.score_proj(rbf_feature))
        if self.use_context_vector:
            if self.context_merge_method == 'vector_concat':
                rbf_feature = torch.cat((rbf_feature, context_feature), dim=1)
                score = self.score_proj(rbf_feature)
            elif self.context_merge_method == 'score_add':
                score = self.score_proj(rbf_feature) + self.context_score_proj(context_feature)
            else:
                raise ValueError('mode error')
        else:
            score = self.score_proj(rbf_feature)
        return score
