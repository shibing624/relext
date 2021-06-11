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
            "中国的首都北京",
            "中国的首都为北京",
        ]
        outs = []
        for sent in sents:
            words, postags, srl, dep = m.parser.tok_parser(sent)
            print(words, postags, srl, dep)
            m_triple = m._get_event_by_srl(words, postags, srl)
            print(m_triple)
            dep_triple = m._get_svo_by_dep(words, postags, dep)
            print(dep_triple)
            outs.append(m_triple)
        self.assertEqual(outs[0], [])
        self.assertEqual([['中国的首都', '为', '北京']], outs[4])

    def test_deal_with_long_sent(self):
        sents = [
            "南京胖哥曾是铅球运动员",
            "因白色标致牌轿车安全气囊弹开已无法行驶，吉某某即驾现场碰擦的黑色别克牌轿车沿中山东路往东快速逃窜，在洪武路和中山东路路口（第三现场）与多辆汽车发生碰撞，并将两名路人周某某（女，27岁，本市人）、陈某某（女，25岁，本市人）撞伤。",
            "案件发生后，许多热心市民向警方报警并现场协助阻止犯罪，警方对热心群众的见义勇为行为表示敬佩和衷心感谢，南京的平安离不开广大人民群众的共同呵护。",
            "今后，南京市公安机关将按照市委、市政府的部署，加强对各类风险隐患的排查整治，始终保持对违法犯罪行为严打高压态势，不断深化平安南京建设，全力保障人民群众生命财产安全。",
            "据央视报道披露的细节，5月29日21时许，犯罪嫌疑人吉某某驾驶租来的白色轿车，在秦淮区金銮巷将其前妻撞倒在地，现场热心市民阻拦无果后，吉某某再次碾压该女子后逃离现场。",
            "中国国家主席习近平访问韩国，并在首尔大学发表演讲。",
        ]
        outs = []
        for sent in sents:
            words, postags, srl, dep = m.parser.tok_parser(sent)
            print(words, postags, srl, dep)
            m_triple = m._get_event_by_srl(words, postags, srl)
            print(m_triple)
            dep_triple = m._get_svo_by_dep(words, postags, dep)
            print(dep_triple)
            outs.append(m_triple)
            chunks = sent.split('，')
            print("short:")
            for chunk in chunks:
                words, postags, srl, dep = m.parser.tok_parser(chunk)
                m_triple = m._get_event_by_srl(words, postags, srl)
                print(m_triple)
                dep_triple = m._get_svo_by_dep(words, postags, dep)
                print(dep_triple)
        self.assertEqual(outs[0], [['南京胖哥', '是', '铅球运动员']])
        self.assertEqual(outs[1], [['吉某某', '驾', '轿车'], ['碰撞', '发生', '汽车']])
        self.assertEqual(outs[2], [['许多热心市民', '报警', '向警方'], ['警方', '表示', '感谢'], ['南京的平安', '离', '呵护']])
        self.assertEqual(outs[3], [['南京市公安机关', '加强', '整治'], ['南京市公安机关', '整治', '隐患'], ['南京市公安机关', '保持', '态势'],
                                   ['南京市公安机关', '深化', '平安南京建设'], ['南京市公安机关', '保障', '安全']])
        self.assertEqual(outs[4], [['央视报道', '披露', '细节'], ['犯罪嫌疑人吉某某', '驾驶', '轿车'], ['犯罪嫌疑人吉某某', '撞倒', '其前妻'],
                                   ['吉某某', '逃离', '现场']])
        self.assertEqual(outs[5], [['中国国家主席习近平', '访问', '韩国'], ['中国国家主席习近平', '发表', '演讲']])


if __name__ == '__main__':
    unittest.main()
