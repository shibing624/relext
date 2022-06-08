# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

from typing import List, Dict, Union
from relext.uie_predictor import UIEPredictor


class InformationExtraction:
    def __init__(self, model_name_or_path='uie-base', schema=(), max_seq_len=512, position_prob=0.5, device='cpu'):
        self.predictor = UIEPredictor(model_name_or_path, schema, max_seq_len, position_prob, device)

    def extract(self, texts: Union[str, List[str]], schema: Union[str, List, Dict]) -> List[Dict]:
        """
        信息抽取
        :param texts: list of str, 文本列表
        :param schema: list of str, 模式列表
        :return:
        """
        self.predictor.set_schema(schema)
        if not self.predictor.schema_tree.has_child():
            raise ValueError('Schema set error. Please check your schema.')
        outputs = self.predictor.predict(texts)

        return outputs
