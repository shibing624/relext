# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import unittest
from pprint import pprint
from loguru import logger
import sys

sys.path.append('..')

from relext import InformationExtraction

m = InformationExtraction()


class TaskTestCase(unittest.TestCase):
    def test_extract(self):
        """测试文本extract triples结果"""
        from paddlenlp import Taskflow
        schema = ['出发地', '目的地', '费用', '时间']
        my_ie = Taskflow("information_extraction", schema=schema)
        r = my_ie("城市内交通费7月5日金额114广州至佛山")
        pprint(r)
        self.assertTrue(r is not None)

    def test_has_schema(self):
        schema = ['时间', '选手', '赛事名称']  # Define the schema for entity extraction
        texts = ["2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"]
        outputs = m.extract(texts, schema)
        pprint(outputs[0])
        self.assertTrue(outputs[0] is not None)

        schema = ['出发地', '目的地', '费用', '时间']
        r = m.extract("城市内交通费7月5日金额114广州至佛山", schema)
        pprint(r)
        self.assertTrue(r is not None)

    def test_no_schema(self):
        texts = ["2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"]
        try:
            outputs = m.extract(texts, [])
        except Exception as e:
            logger.error(e)
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
