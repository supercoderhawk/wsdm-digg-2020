# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from wsdm_digg.constants import *
from wsdm_digg.reranking.plm_knrm import RbfKernelList, RbfKernel


class PlmConvKnrm(nn.Module):
    model_name = 'conv-knrm'

    def __init__(self, args):
        super().__init__()
        self.args = args
        plm_model_name = self.args.plm_model_name
        if plm_model_name not in MODEL_DICT:
            raise ValueError('model name is not supported.')
        model_info = MODEL_DICT[plm_model_name]
        if 'path' in model_info:
            plm_model_name = model_info['path']
        self.plm_model = model_info['model_class'].from_pretrained(plm_model_name)
        if torch.cuda.is_available():
            self.plm_model.cuda()
        self.mean_list = self.args.mean_list
        self.stddev_list = self.args.stddev_list
        assert len(self.mean_list) == len(self.stddev_list)
        self.conv_list = nn.ModuleList()
        self.window_count = len(self.args.window_size_list)
        for window in self.args.window_size_list:
            # print(window)
            # w2p = {1: 0, 2: 0, 3: 1}
            conv = nn.Conv1d(in_channels=self.args.dim_size,
                             out_channels=self.args.filter_size,
                             kernel_size=window, padding=window // 2)
            self.conv_list.append(conv)
        self.kernel = RbfKernelList(self.mean_list, self.stddev_list)
        self.query_max_len = self.args.query_max_len
        # self.doc_max_len = self.args.max_len - self.query_max_len - self.args.special_token_count
        self.doc_max_len = self.args.doc_max_len
        self.use_context_vector = self.args.use_context_vector
        feature_size = len(self.mean_list) * self.window_count ** 2
        if self.use_context_vector:
            self.score_dim = feature_size + self.args.dim_size
        else:
            self.score_dim = feature_size
        self.feature_norm = nn.LayerNorm(feature_size)
        self.score_proj = nn.Linear(self.score_dim, 1, bias=True)
        # self.score_proj = nn.Linear(len(self.mean_list), 1, bias=True)

    def forward(self, token_ids, segment_ids, token_mask, query_lens, doc_lens):
        # token_ids, segment_ids, token_mask, query_lens, doc_lens
        #
        batch_size = query_lens.size(0)
        query_embed = self.plm_model(input_ids=token_ids[0],
                                     attention_mask=token_mask[0],
                                     token_type_ids=segment_ids[0])[0]
        query_embed = query_embed[:, 1:self.query_max_len + 1]
        doc_embed = self.plm_model(input_ids=token_ids[1],
                                   attention_mask=token_mask[1],
                                   token_type_ids=segment_ids[1])[0]
        doc_embed = doc_embed[:, 1:self.doc_max_len + 1]

        contextualized_embed = self.plm_model(input_ids=token_ids,
                                              attention_mask=token_mask,
                                              token_type_ids=segment_ids)[0]
        query_embed = contextualized_embed[:, 1:self.query_max_len + 1]
        doc_start_idx = self.query_max_len + 2
        doc_end_idx = doc_start_idx + self.doc_max_len
        doc_embed = contextualized_embed[:, doc_start_idx:doc_end_idx]
        context_vector = contextualized_embed[:, 0]
        total_rbf_feature = None

        q_range = torch.arange(self.query_max_len).unsqueeze(0).repeat(batch_size, 1)
        if torch.cuda.is_available():
            q_range = q_range.cuda()
        q_mask = q_range >= query_lens.unsqueeze(1)

        q_conv_list, d_conv_list = self.get_conv_matrix(query_embed, doc_embed, query_lens, doc_lens)
        for q_idx in range(self.window_count):
            for d_idx in range(self.window_count):
                q_conv = q_conv_list[q_idx]
                d_conv = d_conv_list[d_idx]
                sim_matrix = self.get_similarity_matrix(q_conv, d_conv, query_lens, doc_lens)
                rbf_feature = self.kernel(sim_matrix, q_mask)
                if total_rbf_feature is None:
                    total_rbf_feature = rbf_feature
                else:
                    total_rbf_feature = torch.cat([total_rbf_feature, rbf_feature], dim=1)
        total_rbf_feature = self.feature_norm(total_rbf_feature)
        score = self.get_rank_score(total_rbf_feature, context_vector)
        return score

    def get_conv_matrix(self, query_embed, doc_embed, query_lens, doc_lens):
        query_conv_result = []
        doc_conv_result = []
        for conv in self.conv_list:
            q_conv = torch.relu(conv(query_embed.permute(0, 2, 1))).permute(0, 2, 1)
            d_conv = torch.relu(conv(doc_embed.permute(0, 2, 1))).permute(0, 2, 1)
            if d_conv.size(1) > self.doc_max_len:
                d_conv = d_conv[:, 1:]
            if q_conv.size(1) > self.query_max_len:
                q_conv = q_conv[:, 1:]
            query_conv_result.append(q_conv)
            doc_conv_result.append(d_conv)

        return query_conv_result, doc_conv_result

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

    def get_rank_score(self, rbf_feature, context_feature):
        if self.use_context_vector:
            score = self.score_proj(torch.cat(rbf_feature, context_feature))
        else:
            score = self.score_proj(rbf_feature)
        score = torch.tanh(score)
        return score
