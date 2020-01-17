# -*- coding: UTF-8 -*-
import argparse


def parse_args(args=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("-exp_name", required=False, type=str, help='')
    parser.add_argument("-train_filename", required=False, type=str, help='')
    parser.add_argument("-test_filename", required=False, type=str, help='')
    parser.add_argument("-search_filename", required=False, type=str, help='')
    parser.add_argument("-dest_base_dir", required=False, type=str, help='')
    parser.add_argument("-batch_size", type=int, default=4, help='')
    parser.add_argument("-query_field", type=str, default='cites_text',
                        choices=['cites_text', 'description_text'], help='')

    parser.add_argument("-epoch", type=int, default=10, help='')
    parser.add_argument("-plm_learning_rate", type=float, default=1e-5, help='')
    parser.add_argument("-rank_learning_rate", type=float, default=1e-3, help='')
    parser.add_argument("-separate_learning_rate", action='store_true', help='')
    parser.add_argument("-save_model_step", type=int, default=2000, help='')
    parser.add_argument("-gradient_accumulate_step", type=int, default=4, help='')
    parser.add_argument("-scheduler_lr", action='store_true', help='')
    parser.add_argument("-scheduler_step", type=int, default=10000, help='')
    parser.add_argument("-scheduler_gamma", type=float, default=0.5, help='')
    parser.add_argument("-lazy_loading", action='store_true', help='')
    parser.add_argument("-separate_query_doc", action='store_true', help='')

    # model parameter
    parser.add_argument("-plm_model_name", required=False, type=str, help='')
    parser.add_argument("-rerank_model_name", required=False, type=str,
                        choices=['plm', 'knrm', 'conv-knrm','mp','pairwise'], help='')
    parser.add_argument("-max_len", type=int, default=512, help='')
    parser.add_argument("-dim_size", type=int, default=768, help='')
    parser.add_argument("-query_max_len", type=int, default=100, help='')
    parser.add_argument("-doc_max_len", type=int, default=500, help='')
    parser.add_argument("-special_token_count", type=int, default=2, choices=[2, 3], help='')
    parser.add_argument("-use_context_vector", action='store_true', help='')
    parser.add_argument("-mean_list", type=float, metavar='N', nargs='+', help='')
    parser.add_argument("-stddev_list", type=float, metavar='N', nargs='+', help='')
    parser.add_argument("-window_size_list", type=int, metavar='N', nargs='+', help='')
    parser.add_argument("-filter_size", type=int, default=128, help='')

    args = parser.parse_args(args)
    return args
