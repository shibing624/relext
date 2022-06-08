# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')
from relext import RelationExtraction

m = RelationExtraction()


class BaseTestCase(unittest.TestCase):
    def test_extract(self):
        """测试文本extract triples结果"""
        a = """
        2021年transformers为生产环境带来次世代最先进的多语种NLP技术。姚明是李秋平的徒弟。
        阿婆主来到立方庭参观公司。阿婆主来到北京立方庭参观自然语义科技公司。
        萨哈夫说，伊拉克将同联合国继续保持合作。 i dont know. do you? 这是 啥？
        """
        t = m.extract(a)
        print('len triple_dict,', len(t))
        print(t)
        self.assertEqual(7, len(t))

    def test_single_sent_extract(self):
        """测试single_sent_extract"""
        a = '阿婆主来到立方庭参观公司'
        t = m.extract(a)
        print('len triple_dict,', len(t))
        print(t)
        self.assertEqual(2, len(t))

    def test_empty_extract(self):
        """测试empty"""
        a = [' ', '', '    ']
        for i in a:
            t = m.extract(i)
            print('len triple_dict,', len(t))
            print(t)
            self.assertEqual(0, len(t))


if __name__ == '__main__':
    unittest.main()
