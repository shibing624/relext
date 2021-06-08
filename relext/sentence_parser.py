# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), lhy<lhy_in_blcu@126.com
@description: LTP parser
"""
from ltp import LTP

from relext.utils.log import logger


class LtpParser:
    def __init__(self):
        self.ltp = LTP()
        logger.debug('use LTP parser.')

    def _get_dep(self, words, postags, sentence):
        """
        依存关系
        :param words:
        :param postags:
        :param sentence:
        :return:
        """
        seg, h = self.ltp.seg([sentence])
        dep = self.ltp.dep(h)
        arcs = dep[0]

        words = ['Root'] + words
        postags = ['w'] + postags
        tuples = list()
        for index in range(len(words) - 1):
            arc_index = arcs[index][0]
            arc_relation = arcs[index][2]
            tuples.append(
                [index + 1, words[index + 1], postags[index + 1], words[arc_index], postags[arc_index], arc_index,
                 arc_relation])
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
                    if arc[-1] in child_dict:
                        child_dict[arc[-1]].append(arc)
                    else:
                        child_dict[arc[-1]] = []
                        child_dict[arc[-1]].append(arc)
            child_dict_list.append([word, postags[index], index, child_dict])
        return child_dict_list

    def parser_syntax(self, words, postags, sentence):
        """
        提取句子依存关系，保留依存结构
        :param words:
        :param postags:
        :param sentence:
        :return:
        """
        tuples = self._get_dep(words, postags, sentence)
        child_dict_list = self._build_parse_child_dict(words, postags, tuples)
        return tuples, child_dict_list

    def seg_pos(self, sentence):
        """
        句子分词及词性标注
        :param sentence:
        :return:
        """
        seg, h = self.ltp.seg([sentence])
        words = seg[0]
        pos = self.ltp.pos(h)
        postags = pos[0]
        return words, postags
