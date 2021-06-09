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


class BaseTestCase(unittest.TestCase):
    def test_extract(self):
        """测试文本extract triples结果"""
        a = """
        2021年transformers为生产环境带来次世代最先进的多语种NLP技术。姚明是李秋平的徒弟。
        阿婆主来到立方庭参观公司。阿婆主来到北京立方庭参观自然语义科技公司。
        萨哈夫说，伊拉克将同联合国继续保持合作。 i dont know. do you? 这是 啥？
        """
        t = m.extract_triples(a)
        print('len triple_dict,', len(t))
        print(t)
        self.assertEqual(len(t), 6)

    def test_single_sent_extract(self):
        """测试single_sent_extract"""
        a = '阿婆主来到立方庭参观公司'
        t = m.extract_triples(a)
        print('len triple_dict,', len(t))
        print(t)
        self.assertEqual(len(t), 2)


if __name__ == '__main__':
    unittest.main()
