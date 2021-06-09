# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from relext.relation_extract import RelationExtract

article = """
    我送她一朵花，她请我吃了一顿饭。2021年transformers为生产环境带来次世代最先进的多语种NLP技术。姚明是李秋平的徒弟。
    阿婆主来到立方庭参观公司。阿婆主来到北京立方庭参观自然语义科技公司。
    萨哈夫说，伊拉克将同联合国继续保持合作。 i dont know. do you? 这是 啥？
          """

if __name__ == '__main__':
    m = RelationExtract()
    out = m.extract_triples(article)
    print(out)
