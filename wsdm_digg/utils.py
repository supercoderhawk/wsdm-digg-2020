# -*- coding: UTF-8 -*-
import os
from pysenal import read_jsonline_lazy, append_line


def result_format(src_filename, dest_filename=None):
    if dest_filename is None:
        dest_filename = os.path.splitext(src_filename)[0] + '.csv'
    for item in read_jsonline_lazy(src_filename):
        desc_id = item['description_id']
        paper_ids = item['docs'][:3]
        if not paper_ids:
            raise ValueError('result is empty')
        line = desc_id + '\t' + '\t'.join(paper_ids)
        append_line(dest_filename, line)
