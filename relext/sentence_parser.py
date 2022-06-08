# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), lhy<lhy_in_blcu@126.com
@description: hanlp代替LTP，NER和dep识别效果更好
"""

from loguru import logger


class SentenceParser:
    def __init__(self):
        import hanlp
        self.model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)  # 世界最大中文语料库
        logger.debug('Use hanlp parser.')

    def tok_parser(self, sentence):
        """
        执行粗颗粒度分词、词性标注和依存句法分析，语义角色标注
        :param sentence:
        :return:
        {
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
            "srl": [
                [["阿婆主", "ARG0", 0, 1], ["来到", "PRED", 1, 2], ["北京立方庭", "ARG1", 2, 4]],
                [["阿婆主", "ARG0", 0, 1], ["参观", "PRED", 4, 5], ["自然语义科技公司", "ARG1", 5, 9]]
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
        """
        if not sentence:
            return
        ret_dict = self.model(sentence, tasks=['tok/coarse', 'pos/pku', 'srl', 'dep'], skip_tasks='tok/fine')
        return ret_dict['tok/coarse'], ret_dict['pos/pku'], ret_dict['srl'], ret_dict['dep']

    def _get_dep(self, words, postags, dep):
        """
        句法依存关系
        :param words:
        :param postags:
        :return:
        """
        tuples = []
        for index in range(len(words)):
            dep_index = dep[index][0]
            dep_index = dep_index - 1 if dep_index > 0 else dep_index
            dep_relation = dep[index][1]
            tuples.append(
                [index, words[index], postags[index], words[dep_index], postags[dep_index], dep_index, dep_relation])
        return tuples

    def _build_parse_child_dict(self, words, postags, tuples):
        """
        为句子中的每个词语维护一个保存句法依存儿子节点的字典
        :param words:
        :param postags:
        :param tuples:
        :return:
        """
        child_dict_list = []
        for index, word in enumerate(words):
            child_dict = dict()
            for arc in tuples:
                if arc[3] == word:
                    if arc[6] in child_dict:
                        child_dict[arc[6]].append(arc)
                    else:
                        child_dict[arc[6]] = []
                        child_dict[arc[6]].append(arc)
            child_dict_list.append([word, postags[index], index, child_dict])
        return child_dict_list

    def parser_syntax(self, words, postags, dep):
        """
        提取句子依存关系，保留依存结构
        :param words:
        :param postags:
        :return:
        """
        tuples = self._get_dep(words, postags, dep)
        child_dict_list = self._build_parse_child_dict(words, postags, tuples)
        return tuples, child_dict_list
