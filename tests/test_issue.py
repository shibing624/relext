# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

sys.path.append('..')


class IssueTestCase(unittest.TestCase):

    def test_sim_diff(self):
        a = '我送她一朵花，她请我吃了一顿饭。'
        self.assertTrue(len(a) > 0)


if __name__ == '__main__':
    unittest.main()
