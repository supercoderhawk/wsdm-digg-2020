# -*- coding: UTF-8 -*-
import requests
from wsdm_digg.constants import ES_API_URL


def get_paper(paper_id):
    """
    get paper content from elastic search index
    :param paper_id:
    :return:
    """
    url = ES_API_URL + '/_doc/{}'.format(paper_id)
    ret = requests.get(url)
    if ret.ok:
        paper = ret.json()['_source']
        return paper
