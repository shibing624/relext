# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')


class SegTestCase(unittest.TestCase):
    def test_hanlp(self):
        """测试文本segment结果"""
        import hanlp
        HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)  # 世界最大中文语料库
        d = HanLP(['阿婆主来到北京立方庭参观自然语义科技公司。'])
        print(d)
        d.pretty_print()

        HanLP('阿婆主来到北京立方庭参观自然语义科技公司。', tasks='tok').pretty_print()
        HanLP('阿婆主来到北京立方庭参观自然语义科技公司。', tasks='tok/coarse').pretty_print()
        HanLP('阿婆主来到北京立方庭参观自然语义科技公司。', tasks='pos/pku').pretty_print()
        # 执行粗颗粒度分词和PKU词性标注
        HanLP('阿婆主来到北京立方庭参观自然语义科技公司。', tasks=['tok/coarse', 'pos/pku'], skip_tasks='tok/fine').pretty_print()

        # 执行分词和MSRA标准NER
        HanLP('阿婆主来到北京立方庭参观自然语义科技公司。', tasks='ner/msra').pretty_print()
        # 执行分词、词性标注和依存句法分析
        doc = HanLP('阿婆主来到北京立方庭参观自然语义科技公司。', tasks=['pos', 'dep'])
        doc.pretty_print()
        print('dep:', doc)

        # 执行分词、词性标注和短语成分分析
        doc = HanLP('阿婆主来到北京立方庭参观自然语义科技公司。', tasks=['pos', 'con'])
        doc.pretty_print()

        # 执行粗颗粒度分词、词性标注和依存句法分析
        doc = HanLP('阿婆主来到北京立方庭参观自然语义科技公司。', tasks=['tok/coarse', 'pos/pku', 'dep'], skip_tasks='tok/fine')
        doc.pretty_print()
        print('all:', doc)

    def test_hanlp_dep(self):
        """测试文本segment dep结果"""
        import hanlp
        HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)  # 世界最大中文语料库
        # 执行粗颗粒度分词、词性标注和依存句法分析
        doc = HanLP('阿婆主来到北京立方庭参观自然语义科技公司。', tasks=['tok/coarse', 'pos/pku', 'dep'], skip_tasks='tok/fine')
        exp = {
            "tok/coarse": [
                "阿婆主",
                "来到",
                "北京立方庭",
                "参观",
                "自然语义科技公司",
                "。"
            ],
            "pos/pku": [
                "n",
                "v",
                "ns",
                "v",
                "n",
                "w"
            ],
            "dep": [
                (2, "nsubj"),
                (0, "root"),
                (2, "dobj"),
                (2, "conj"),
                (4, "dobj"),
                (2, "punct")
            ]
        }
        print(doc)
        self.assertEqual(doc, exp)

    def test_jieba_seg(self):
        """test jieba"""
        import jieba
        a = "针对网友对荣荣所选择的武警北京二院的治疗效果及其内部管理问题的质疑"
        b = jieba.lcut(a, cut_all=False)
        print('cut_all=False', b)

        b = jieba.lcut(a, cut_all=True)
        print('cut_all=True', b)

        b = jieba.lcut(a, HMM=True)
        print('HMM=True', b)

        b = jieba.lcut(a, HMM=False)
        print('HMM=False', b)

    def test_ltp_seg(self):
        """test ltp"""
        from ltp import LTP
        ltp = LTP()
        a = "针对网友对荣荣所选择的武警北京二院的治疗效果及其内部管理问题的质疑"
        seg, hidden = ltp.seg([a])
        print(a, seg)

    def test_hanlp_seg(self):
        """hanlp seg结果"""
        import hanlp
        HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)  # 世界最大中文语料库
        out = HanLP('针对网友对荣荣所选择的武警北京二院的治疗效果及其内部管理问题的质疑。', tasks='tok')
        print(out)
        out = HanLP('针对网友对荣荣所选择的武警北京二院的治疗效果及其内部管理问题的质疑。', tasks='tok/coarse')
        out.pretty_print()

        out = HanLP('国家主席习近平访问韩国', task='tok/coarse')
        out.pretty_print()


if __name__ == '__main__':
    unittest.main()
