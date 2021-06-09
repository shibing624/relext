# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')
from relext.relation_extract import RelationExtract

m = RelationExtract()


class RelTestCase(unittest.TestCase):
    def test_triples_by_dep(self):
        """测试文本_get_triples_by_dep结果"""
        sents = [
            "我送莉莉一朵花，",
            "我送莉莉一朵玫瑰花，",
            "莉莉请我吃了一顿饭。",
            "2021年transformers为生产环境带来次世代最先进的多语种NLP技术。",
            "姚明是李秋平的徒弟。",
            "阿婆主来到立方庭参观公司。",
            "i dont know. do you? 这是 啥",
        ]
        outs = []
        for sent in sents:
            words, postags = m.parser.seg_pos(sent)
            m_triple = m._get_triples_by_dep(words, postags, sent)
            print(m_triple)
            outs.append(m_triple)
        self.assertTrue(outs[0], [['我', '送', '莉莉']])
        self.assertTrue(outs[1], [['我', '送', '玫瑰花']])
        self.assertTrue(outs[2], [['莉莉', '请', '吃']])
        self.assertTrue(outs[3], [['姚明', '是', '徒弟']])
        self.assertTrue(outs[4], [['阿婆主', '来到', '立方庭']])
        self.assertTrue(outs[5], [['这', '是', '啥']])


if __name__ == '__main__':
    unittest.main()
