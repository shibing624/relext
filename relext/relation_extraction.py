# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), lhy<lhy_in_blcu@126.com
@description:

refer: https://github.com/liuhuanyong/TextGrapher/text_grapher.py
"""
import os
import re
from collections import Counter
from loguru import logger

from relext.graph import Graph
from relext.keywords_textrank import TextKeyword
from relext.sentence_parser import SentenceParser

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

default_ner_dict = {
    "nr": "人名",
    'ns': '地名',
    'nt': '机构名',
}


class RelationExtraction:
    def __init__(self, ner_dict=None):
        self.text_keyword = TextKeyword()
        self.parser = SentenceParser()
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
            if pos in self.ner_dict:
                ners.append(words[index] + '/' + pos)
        return ners

    def _get_coexist(self, wordpos_sents, ners):
        """
        构建实体之间的共现关系
        :param wordpos_sents: 带实体的句子
        :param ners: 实体词
        :return:
        """
        co_list = []
        for sent in wordpos_sents:
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

    @staticmethod
    def _search_attr(verb, child_dict_list, attr_list=None):
        """
        根据谓语动词找属性词
        :param verb:
        :param child_dict_list:
        :return:
        """
        if attr_list is None:
            attr_list = ['attr']
        objs = []
        for child in child_dict_list:
            wd = child[0]
            attr = child[3]
            if wd == verb:
                for i in attr_list:
                    if i not in attr:
                        continue
                    vob = attr['attr'][0]
                    objs.append(vob[1])
        return objs

    @staticmethod
    def _search_obj(verb, child_dict_list, obj_list=None):
        """
        根据谓语动词找宾语词
        :param verb:
        :param child_dict_list:
        :return:
        """
        if obj_list is None:
            obj_list = ['dobj', 'pobj', 'lobj', 'range']
        objs = []
        for child in child_dict_list:
            wd = child[0]
            attr = child[3]
            if wd == verb:
                for i in obj_list:
                    if i not in attr:
                        continue
                    vob = attr[i][0]
                    objs.append(vob[1])
        return objs

    def _get_svo_by_dep(self, words, postags, dep, subj_list=None, verb_list=None):
        """
        抽取出关系三元组，提取主语 - 动词 - 对象三元组（subject，verb，object；主谓宾）
        :param words:
        :param postags:
        :param subj_list: 主语，句法类型，参考 https://github.com/shibing624/relext/blob/main/docs/dep_sd_zh.md
        :param verb_list: 谓语，动词词性，参考 https://github.com/shibing624/relext/blob/main/docs/pos_pku.md
        :return:
        """
        svo = []
        if verb_list is None:
            # 动词
            verb_list = ['v']
        if subj_list is None:
            # 主语和主题词
            subj_list = ['nsubj', 'xsubj', 'nsubjpass', 'top']
        tuples, child_dict_list = self.parser.parser_syntax(words, postags, dep)
        for tuple in tuples:
            v_pos = tuple[4]
            rel = tuple[6]
            # 查询主谓结果
            if rel in subj_list and v_pos in verb_list:
                sub_wd = tuple[1]
                verb_wd = tuple[3]
                # 补充宾语
                attrs = self._search_attr(verb_wd, child_dict_list)
                objs = self._search_obj(verb_wd, child_dict_list)
                subj = sub_wd
                verb = verb_wd
                for attr in attrs:
                    svo.append([subj, verb, attr])
                for obj in objs:
                    svo.append([subj, verb, obj])
        return svo

    def _get_event_by_srl(self, words, postags, srls, subj_list=None, verb_list=None, obj_list=None):
        """
        抽取出施事者，谓词，受事者三元组
        :param words:
        :param postags:
        :param srls: 语义角色标注类型，参考  https://github.com/shibing624/relext/blob/main/docs/srl_cpb.md
        "srl": [
                [["阿婆主", "ARG0", 0, 1], ["来到", "PRED", 1, 2], ["北京立方庭", "ARG1", 2, 4]],
                [["阿婆主", "ARG0", 0, 1], ["参观", "PRED", 4, 5], ["自然语义科技公司", "ARG1", 5, 9]]
            ],
        :return:
        """
        ret = []
        # 谓语动词
        if verb_list is None:
            verb_list = ['PRED']
        # 施事者
        if subj_list is None:
            subj_list = ['ARG0']
        # 受事者
        if obj_list is None:
            obj_list = ['ARG1']
        words_list = [[i[0], i[1]] for i in zip(words, postags)]
        for srl in srls:
            if len(srl) >= 3:
                subj = ''
                verb = ''
                obj = ''
                for srl_unit in srl:
                    # srl_unit format: ["阿婆主", "ARG0", 0, 1]
                    word, rel, begin, end = srl_unit[0], srl_unit[1], srl_unit[2], srl_unit[3]
                    if rel in subj_list:
                        subj = self._get_last_word(words_list, srl_unit)
                    elif rel in verb_list:
                        verb = word
                    elif rel in obj_list:
                        obj = self._get_last_word(words_list, srl_unit)
                if subj and verb and obj:
                    ret.append([subj, verb, obj])
        return ret

    @staticmethod
    def _get_last_word(words_list, triple):
        """
        取多词组合的长词中的最后词
        :param words_list:
        :param triple:
        :return: word
        """
        wd, begin, end = triple[0], triple[2], triple[3]
        words = words_list[begin:end]
        if len(words) == 0:
            word = ''
        elif len(words) in [1, 2, 3]:
            word = wd
        else:
            word = words[-1][0]  # 重点词在文本靠后
        return word

    @staticmethod
    def _get_entity_relation(ners, keywords, subsent_segs):
        """
        通过关键词与实体进行实体关系抽取
        :param ners: 实体
        :param keywords: 关键词
        :param subsent_segs: 句子，分词过
        :return:
        """
        rels = []
        relation_word_pairs = []
        ners = [i.split('/')[0] for i in set(ners)]
        for sent_seg in subsent_segs:
            tmp = []
            for wd in sent_seg:
                if wd in ners + keywords:
                    tmp.append(wd)
            if len(tmp) > 1:
                relation_word_pairs.append(tmp)
        for ner in ners:
            for pair in relation_word_pairs:
                if ner in pair:
                    tmp = ['->'.join([ner, wd]) for wd in pair if
                           wd in keywords and wd != ner and len(wd) > 1 and len(ner) > 1]
                    if tmp:
                        rels += tmp
        return rels

    @staticmethod
    def show_triples(triple_dict, html_path="graph_show.html"):
        # 保存抽取的实体关系，即三元组
        triple_list = []
        for k, v in triple_dict.items():
            for i in v:
                if i not in triple_list:
                    triple_list.append(i)
        graph = Graph(triple_list)
        graph.show(html_path)
        logger.debug("save to graph done.")

    @staticmethod
    def seg_to_sentence(text, regex=r'[？?！!。；;：:\n\r\t ]'):
        """
        利用标点符号，将文章进行句子切分处理，保留逗号短句
        :param regex:
        :param text: article
        :return:
        """
        return [sentence.strip() for sentence in re.split(regex, text) if sentence.strip()]

    def extract(self, text, num_keywords=10):
        """
        三元组抽取
        :param text:
        :param num_keywords:
        :return:
        """
        # 存储实体关系抽取结果，三元组
        triple_dict = {}
        # 对文章进行去噪处理
        text = self.remove_noisy(text)
        # 对文章进行短句切分处理
        sents = self.seg_to_sentence(text)
        sents_seg = []
        # 存储整篇文章的词频信息
        words_list = []
        # 保存具有命名实体的句子
        wordpos_sents = []
        # 保存命名实体
        ners = []
        # 保存主谓宾
        svos = []
        # 事件三元组
        events = []
        for sent in sents:
            sent = sent.strip()
            if not sent:
                continue
            words, postags, srls, deps = self.parser.tok_parser(sent)
            words_list += [[i[0], i[1]] for i in zip(words, postags)]
            sents_seg.append([i[0] for i in zip(words, postags)])
            m_events = self._get_event_by_srl(words, postags, srls)
            m_ners = self._get_ners(words, postags)
            if m_ners:
                m_svo = self._get_svo_by_dep(words, postags, deps)
                if not m_svo:
                    continue
                svos += m_svo
                ners += m_ners
                events += m_events
                wordpos_sents.append([words, postags])

        # 获取文章关键词
        kw_triples = []
        keywords = [i[0] for i in self.text_keyword.extract_keywords(words_list, num_keywords)]
        for keyword in keywords:
            name = keyword
            cate = ''
            obj = '关键词'
            kw_triples.append([name, cate, obj])
        if kw_triples:
            triple_dict['keyword'] = kw_triples

        # 获取文章词频信息
        freq_triples = []
        word_dict = [i[0] for i in Counter([i[0] for i in words_list
                                            if i[1][0] in ['n', 'v'] and len(i[0]) > 1]).most_common()][:num_keywords]
        for wd in word_dict:
            name = wd
            cate = ''
            obj = '高频词'
            freq_triples.append([name, cate, obj])
        if freq_triples:
            triple_dict['freq'] = freq_triples

        # 获取全文命名实体
        ner_triples = []
        ner_dict = {i[0]: i[1] for i in Counter(ners).most_common()}
        for m_ners in ner_dict:
            name, pos = m_ners.split('/')
            cate = self.ner_dict.get(pos)
            obj = '实体词'
            if len(name) > 1:
                ner_triples.append([name, cate, obj])
        if ner_triples:
            triple_dict['ner'] = ner_triples

        # 获取全文命名实体共现信息，构建实体共现三元组
        coexist_triples = []
        co_dict = self._get_coexist(wordpos_sents, list(ner_dict.keys()))
        for c in co_dict.keys():
            name = c.split('@')[0].split('/')[0]
            cate = ''
            obj = c.split('@')[1].split('/')[0]
            if len(name) > 1 and len(obj) > 1:
                coexist_triples.append([name, cate, obj])
        if coexist_triples:
            triple_dict['coexist'] = coexist_triples

        # 将关键词与实体进行关系抽取
        ner_keyword_triples = []
        entity_rels = self._get_entity_relation(ners, keywords, sents_seg)
        for e in set(entity_rels):
            name = e.split('->')[0]
            cate = ''
            obj = e.split('->')[1]
            if len(name) > 1 and len(obj) > 1:
                ner_keyword_triples.append([name, cate, obj])
        if ner_keyword_triples:
            triple_dict['ner_keyword'] = ner_keyword_triples

        # 主谓宾三元组
        svo_triples = []
        for t in svos:
            name = t[0]
            obj = t[2]
            if len(name) > 1 and len(obj) > 1 and (
                    name in keywords or obj in keywords or
                    name in ner_dict or obj in ner_dict or
                    name in word_dict or obj in word_dict
            ):
                svo_triples.append(t)
        if svo_triples:
            triple_dict['svo'] = svo_triples

        # 获取文章事件(施事者，谓词，受事者)三元组
        event_triples = []
        for t in events:
            name = t[0]
            obj = t[2]
            if len(name) > 1 and len(obj) > 1 and (
                    name in keywords or obj in keywords or
                    name in ner_dict or obj in ner_dict or
                    name in word_dict or obj in word_dict
            ):
                event_triples.append(t)
        if event_triples:
            triple_dict['event'] = event_triples
        return triple_dict
