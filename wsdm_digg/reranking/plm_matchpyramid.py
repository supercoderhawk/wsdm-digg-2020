# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from wsdm_digg.utils import load_plm_model


class PlmMatchPyramid(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.plm_model, self.tokenizer = load_plm_model(self.args.plm_model_name)
        self.query_max_len = self.args.query_max_len
        self.doc_max_len = self.args.max_len - self.query_max_len - self.args.special_token_count
        kernel_count = 100
        # inchanels = [1, kernel_count]
        # out_channels = [kernel_count, kernel_count]
        kernel_size = [3, 3]
        activation = nn.modules.activation.ReLU
        self.conv = self._get_conv_block(1, kernel_count, kernel_size, activation)
        dpool_size = [3, 10]
        self.dpool = nn.AdaptiveAvgPool2d(dpool_size)
        output_dim = dpool_size[0] * dpool_size[1] * kernel_count
        self.output_proj = nn.Linear(output_dim, 1)

    def _get_conv_block(self, in_channel, output_channel, kernel_size, activation):
        conv = nn.Sequential(
            # Same padding
            nn.ConstantPad2d(
                (0, kernel_size[1] - 1, 0, kernel_size[0] - 1), 0
            ),
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=output_channel,
                kernel_size=kernel_size
            ),
            activation
        )
        return conv

    def forward(self, token_ids, segment_ids, token_mask, query_lens, doc_lens):
        contextualized_embed = self.plm_model(input_ids=token_ids,
                                              attention_mask=token_mask,
                                              token_type_ids=segment_ids)[0]
        query_embed = contextualized_embed[:, 1:self.query_max_len + 1]
        doc_start_idx = self.query_max_len + 2
        doc_end_idx = doc_start_idx + self.doc_max_len
        doc_embed = contextualized_embed[:, doc_start_idx:doc_end_idx]
        match_matrix = self.get_match_result(query_embed, doc_embed)
        conv = self.conv(match_matrix)
        embed_pool = self.dpool(conv)
        scores = self.output_proj(torch.flatten(embed_pool, start_dim=1))
        return scores

    def get_match_result(self, query_embed, doc_embed):
        match_matrix = torch.bmm(query_embed, doc_embed.view(0, 2, 1))
        return match_matrix
