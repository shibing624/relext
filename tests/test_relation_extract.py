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
            "我送莉莉玫瑰花，",
            "莉莉请我吃了一顿饭。",
            "姚明是李秋平的徒弟。",
            "阿婆主来到立方庭参观公司。",
            "i dont know. do you? 这是啥",
            "玻尿酸为生产环境带来次世代最先进的多语种NLP技术。",
            "一个中国人"
        ]
        outs = []
        for sent in sents:
            words, postags, srl, dep = m.parser.tok_parser(sent)
            print(words, postags, srl, dep)
            m_triple = m._get_svo_by_dep(words, postags, dep)
            print(m_triple)
            outs.append(m_triple)
        self.assertEqual(outs[0], [['我', '送', '莉莉'], ['我', '送', '花']])
        self.assertEqual(outs[1], [['我', '送', '莉莉'], ['我', '送', '玫瑰花']])  # 我 送 玫瑰花
        self.assertEqual(outs[2], [['莉莉', '请', '我']])
        self.assertEqual(outs[3], [['姚明', '是', '徒弟']])
        self.assertEqual(outs[4], [['阿婆主', '来到', '立方庭']])
        self.assertEqual(outs[5], [['这', '是', '啥']])
        self.assertEqual(outs[6], [['玻尿酸', '带来', '技术']])
        self.assertEqual(outs[7], [])

    def test_triples_by_dep_long_term(self):
        """测试文本_get_triples_by_dep 长词结果"""
        sents = [
            "国家主席习近平访问韩国",
            "在首尔大学发表演讲",
            "中国国家主席习近平在首尔大学发表演讲",
            "武警北京二院应负有主要责任",
            "针对网友对荣荣所选择的武警北京二院的治疗效果及其内部管理问题的质疑",
        ]
        outs = []
        for sent in sents:
            words, postags, srl, dep = m.parser.tok_parser(sent)
            print(words, postags, dep)
            m_triple = m._get_svo_by_dep(words, postags, dep)
            print(m_triple)
            outs.append(m_triple)
        self.assertEqual(outs[0], [['习近平', '访问', '韩国']])
        self.assertEqual(outs[1], [])
        self.assertEqual(outs[2], [['习近平', '发表', '演讲']])
        self.assertEqual(outs[3], [['武警北京二院', '负有', '责任']])
        self.assertEqual(outs[4], [])

    def test_triples_by_srl(self):
        """测试文本_get_triples_by_srl结果"""
        sents = [
            "奥巴马博士毕业于哈弗大学",
            "习近平主席和李克强总理坐飞机访问美国和英国。",
            "习近平对埃及进行国事访问",
            "厦门大学的朱崇实校长来到了北京的五道口学院，",
            "中国国家主席习近平访问韩国，并在首尔大学发表演讲。",
            "《古世》是连载于云中书城的网络小说，作者是未弱",
        ]
        outs = []
        for sent in sents:
            words, postags, srl, dep = m.parser.tok_parser(sent)
            print(words, postags, srl, dep)
            m_triple = m._get_event_by_srl(words, postags, srl)
            print(m_triple)
            outs.append(m_triple)
        self.assertEqual(outs[0], [['奥巴马', '毕业', '哈弗大学']])
        self.assertEqual(outs[1], [['李克强', '访问', '英国']])
        self.assertEqual(outs[2], [['习近平', '进行', '国事访问']])
        self.assertEqual(outs[3], [['朱崇实', '来到', '五道口学院']])
        self.assertEqual(outs[4], [['习近平', '访问', '韩国'], ['习近平', '发表', '演讲']])
        self.assertEqual(outs[5], [['《古世》', '是', '云中'], ['作者', '是', '未弱']])

    def test_triples(self):
        """测试文本_get_triples结果"""
        sents = [
            "曹村辖区总面积约10平方公里，其中耕地132公顷。",
            "硫酸钙较为常用，其性质稳定，无嗅无味，微溶于水",
            "蔡竞，男，汉族，四川射洪人，西南财经大学经济学院毕业，经济学博士。",
            " ",
        ]
        outs = []
        for sent in sents:
            words, postags, srl, dep = m.parser.tok_parser(sent)
            print(words, postags, srl, dep)
            m_triple = m._get_event_by_srl(words, postags, srl)
            print(m_triple)
            outs.append(m_triple)


if __name__ == '__main__':
    unittest.main()
