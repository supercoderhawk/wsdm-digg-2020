# -*- coding: UTF-8 -*-
import argparse


def parse_args(args=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-train_filename', type=str)
    parser.add_argument('-test_filename', type=str)
    parser.add_argument('-exp_name', type=str)
    parser.add_argument('-dest_base_dir', type=str)
    parser.add_argument('-data_type', type=str, choices=['query', 'doc'], default='doc')
    parser.add_argument('-query_field', type=str, choices=['cites_text', 'description_text'],
                        default='cites_text')
    parser.add_argument('-mode', type=str, choices=['train', 'eval', 'inference'])
    parser.add_argument('-batch_size', type=int, default=3)
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-learning_rate', type=float, default=5e-6)
    parser.add_argument('-lazy_loading', action='store_true')
    parser.add_argument('-save_model_step', type=int, default=5000)
    parser.add_argument("-gradient_accumulate_step", type=int, default=5, help='')
    parser.add_argument("-scheduler_lr", action='store_true', help='')
    parser.add_argument("-scheduler_step", type=int, default=10000, help='')
    parser.add_argument("-scheduler_gamma", type=float, default=0.5, help='')
    parser.add_argument("-negative_sample_count", type=int, default=5, help='')

    # model parameters
    parser.add_argument('-plm_model_name', type=str)
    parser.add_argument('-dim_size', type=int, default=768)
    parser.add_argument('-max_len', type=int, default=512)
    parser.add_argument('-embed_mode', type=str, choices=['USE', 'attention', 'CLS'],
                        default='USE')
    parser.add_argument('-use_context_vector', action='store_true')
    args = parser.parse_args(args)
    return args
