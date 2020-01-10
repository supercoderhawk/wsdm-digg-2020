# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from wsdm_digg.utils import load_plm_model


class PlmModel(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()
        self.plm_model, self.tokenizer = load_plm_model(self.args.plm_model_name)
        self.attn_proj = nn.Linear(self.args.dim_size, self.args.dim_size)
        self.vector_merge_proj = nn.Linear(self.args.dim_size * 2, self.args.dim_size)

    def forward(self, batch, prefix):
        if not prefix:
            token_field = 'tokens'
            mask_field = 'masks'
            len_field = 'sent_lens'
        else:
            token_field = '{}_tokens'.format(prefix)
            mask_field = '{}_masks'.format(prefix)
            len_field = '{}_sent_lens'.format(prefix)

        output = self.plm_model(input_ids=batch[token_field],
                                attention_mask=batch[mask_field])[0]

        cls_embed = output[:, 0]
        token_embed = output[:, 1:]
        token_mask = batch[mask_field][:, 1:].unsqueeze(2)
        sent_lens = batch[len_field]

        if self.args.embed_mode == 'USE':
            sent_embed = self.get_USE_embedding(cls_embed, token_embed, token_mask, sent_lens)
        else:
            sent_embed = self.get_attention_embedding(cls_embed, token_embed, token_mask)

        return sent_embed

    def get_attention_embedding(self, cls_embed, token_embed, token_mask):
        """
        attention Model
        :param cls_embed:
        :param token_embed:
        :param token_mask:
        :return:
        """
        attn_weights = torch.bmm(self.attn_proj(cls_embed.unsqueeze(1)), token_embed)
        attn_weights.masked_fill_(token_mask, float('-inf'))
        attn_scores = torch.softmax(attn_weights, dim=-1)
        context_embed = torch.sum(attn_scores * token_embed, dim=1)
        sent_embed = (cls_embed + context_embed) / 2
        return sent_embed

    def get_USE_embedding(self, cls_embed, token_embed, token_mask, sent_lens):
        """
        Universal Sentence Encoder Transformer Model
        :param sent_lens:
        :param token_embed:
        :param token_mask:
        :return:
        """
        token_embed.masked_fill_(token_mask, 0.0)
        factor = torch.sqrt(sent_lens.type(torch.float)).unsqueeze(1)
        sent_embed = token_embed.sum(dim=1) / factor
        # if self.args.use_context_vector:
        #     pass
        return sent_embed
