# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from relext import RelationExtract

article = """
    9月13日，咸阳市公安局在解放路街角捣毁一传销窝点，韩立明抓住主犯姚丽丽立下二等功。彩虹分局西区派出所民警全员出动查处有功。
          """

if __name__ == '__main__':
    m = RelationExtract()
    triples = m.extract_triples(article)
    print(triples)
