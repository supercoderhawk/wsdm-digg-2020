# -*- coding: UTF-8 -*-
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_filename', type=str)
    parser.add_argument('-test_filename', type=str)
    parser.add_argument('-dim_size', type=int)
    parser.add_argument('-embed_mode', type=str, choices=['USE', 'attention'])
    parser.add_argument('-batch_size', type=int, default=4)
    args = parser.parse_args()
    return args
