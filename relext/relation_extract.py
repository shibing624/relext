# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), lhy<lhy_in_blcu@126.com
@description:

refer: https://github.com/liuhuanyong/TextGrapher/text_grapher.py
"""

import re
from collections import Counter

from relext.graph import Graph
from relext.keywords_textrank import TextKeyword
from relext.sentence_parser import LtpParser
from relext.utils.log import logger

default_ner_dict = {
    'nh': '人物',
    "nr": "人名",
    'ni': '机构',
    "nt": "机构团体名",
    'ns': '地名'
}


class RelationExtract:
    def __init__(self, ner_dict={}):
        self.text_keyword = TextKeyword()
        self.parser = LtpParser()
        self.ner_dict = ner_dict if ner_dict else default_ner_dict

    @staticmethod
    def remove_noisy(content):
        """
        移除括号内的信息，去除噪声
        :param content:
        :return:
        """
        p1 = re.compile(r'（[^）]*）')
        p2 = re.compile(r'\([^\)]*\)')
        return p2.sub('', p1.sub('', content))

    def _get_ners(self, words, postags):
        """
        获取命名实体词
        :param words:
        :param postags:
        :return:
        """
        ners = []
        for index, pos in enumerate(postags):
            if pos in self.ner_dict.keys():
                ners.append(words[index] + '/' + pos)
        return ners

    def _get_coexist(self, ner_sents, ners):
        """
        构建实体之间的共现关系
        :param ner_sents: 带实体的句子
        :param ners: 实体词
        :return:
        """
        co_list = []
        for sent in ner_sents:
            words = [i[0] + '/' + i[1] for i in zip(sent[0], sent[1])]
            co_ners = set(ners).intersection(set(words))
            co_info = self.combination(list(co_ners))
            co_list += co_info
        if not co_list:
            return {}
        return {i[0]: i[1] for i in Counter(co_list).most_common()}

    @staticmethod
    def combination(lst):
        """列表全排列"""
        combines = []
        if len(lst) == 0:
            return combines
        for i in lst:
            for j in lst:
                if i == j:
                    continue
                combines.append('@'.join([i, j]))
        return combines

    def _get_triples_by_dep(self, words, postags, sentence):
        """
        抽取出关系三元组，提取主体 - 动词 - 对象三元组（subject，verb，object）
        :param words:
        :param postags:
        :param sentence:
        :return:
        """
        svo = []
        tuples, child_dict_list = self.parser.parser_syntax(words, postags, sentence)
        for tuple in tuples:
            rel = tuple[-1]
            if rel in ['SBV']:
                sub_wd = tuple[1]
                verb_wd = tuple[3]
                obj = self._complete_VOB(verb_wd, child_dict_list)
                subj = sub_wd
                verb = verb_wd
                if not obj:
                    svo.append([subj, verb])
                else:
                    svo.append([subj, verb + obj])
        return svo

    def _filter_triples(self, triples, ners):
        """
        过滤出跟命名实体相关的事件三元组
        :param triples:
        :param ners:
        :return:
        """
        ner_triples = []
        for ner in ners:
            for triple in triples:
                if ner in triple:
                    ner_triples.append(triple)
        return ner_triples

    def _complete_VOB(self, verb, child_dict_list):
        """
        根据SBV找VOB
        :param verb:
        :param child_dict_list:
        :return:
        """
        for child in child_dict_list:
            wd = child[0]
            attr = child[3]
            if wd == verb:
                if 'VOB' not in attr:
                    continue
                vob = attr['VOB'][0]
                obj = vob[1]
                return obj
        return ''

    def _get_rel_entity(self, ners, keywords, subsent_segs):
        """
        通过关键词与实体进行实体关系抽取
        :param ners: 实体
        :param keywords: 关键词
        :param subsent_segs: 句子，分词过
        :return:
        """
        events = []
        rels = []
        sents = []
        ners = [i.split('/')[0] for i in set(ners)]
        keywords = [i[0] for i in keywords]
        for sent in subsent_segs:
            tmp = []
            for wd in sent:
                if wd in ners + keywords:
                    tmp.append(wd)
            if len(tmp) > 1:
                sents.append(tmp)
        for ner in ners:
            for sent in sents:
                if ner in sent:
                    tmp = ['->'.join([ner, wd]) for wd in sent if wd in keywords and wd != ner and len(wd) > 1]
                    if tmp:
                        rels += tmp
        for e in set(rels):
            events.append([e.split('->')[0], e.split('->')[1]])
        return events

    @staticmethod
    def seg_to_sentence(text):
        """
        利用标点符号，将文章进行短句切分处理
        :param text: article
        :return:
        """
        return [sentence for sentence in re.split(r'[，,？?！!。；;：:\n\r\t ]', text) if sentence]

    def extract_triples(self, text, num_keywords=10):
        """
        三元组抽取
        :param text:
        :param num_keywords:
        :return:
        """
        if not text:
            return
        # 对文章进行去噪处理
        text = self.remove_noisy(text)
        # 对文章进行短句切分处理
        subsents = self.seg_to_sentence(text)
        subsents_seg = []
        # words_list存储整篇文章的词频信息
        words_list = []
        # ner_sents保存具有命名实体的句子
        ner_sents = []
        # ners保存命名实体
        ners = []
        # triples保存主谓宾短语
        triples = []
        # 存储文章事件
        events = []
        for sent in subsents:
            words, postags = self.parser.seg_pos(sent)
            words_list += [[i[0], i[1]] for i in zip(words, postags)]
            subsents_seg.append([i[0] for i in zip(words, postags)])
            m_ners = self._get_ners(words, postags)
            if m_ners:
                m_triple = self._get_triples_by_dep(words, postags, sent)
                if not m_triple:
                    continue
                triples += m_triple
                ners += m_ners
                ner_sents.append([words, postags])

        # 获取文章关键词, 并组织图谱
        keywords = [i[0] for i in self.text_keyword.extract_keywords(words_list, num_keywords)]
        for keyword in keywords:
            name = keyword
            cate = '关键词'
            events.append([name, cate])
        # 对三元组进行event构建，这个可以做
        for t in triples:
            if (t[0] in keywords or t[1] in keywords) and len(t[0]) > 1 and len(t[1]) > 1:
                events.append([t[0], t[1]])

        # 获取文章词频信息
        word_dict = [i for i in
                     Counter([i[0] for i in words_list if i[1][0] in ['n', 'v'] and len(i[0]) > 1]).most_common()][:10]
        for wd in word_dict:
            name = wd[0]
            cate = '高频词'
            events.append([name, cate])

        # 获取全文命名实体
        ner_dict = {i[0]: i[1] for i in Counter(ners).most_common()}
        for m_ners in ner_dict:
            name = m_ners.split('/')[0]
            cate = self.ner_dict[m_ners.split('/')[1]]
            events.append([name, cate])

        # 获取全文命名实体共现信息,构建事件共现网络
        co_dict = self._get_coexist(ner_sents, list(ner_dict.keys()))
        co_events = [[i.split('@')[0].split('/')[0], i.split('@')[1].split('/')[0]] for i in co_dict]
        events += co_events
        # 将关键词与实体进行关系抽取
        events_entity_keyword = self._get_rel_entity(ners, keywords, subsents_seg)
        events += events_entity_keyword
        print(events)
        graph = Graph(events)
        graph.save_graph("graph_saved.html")
        logger.debug("save to graph done.")
        return graph
