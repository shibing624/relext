# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import re
from multiprocessing import cpu_count

import numpy as np
import paddle
from loguru import logger
from paddle.utils.download import get_path_from_url
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.utils.tools import get_bool_ids_greater_than, get_span

from relext.uie_model import UIE
from relext.utils import MODEL_MAP, USER_DATA_DIR


class InferBackend:
    def __init__(self,
                 model_dir,
                 device='cpu'):
        self._num_threads = math.ceil(cpu_count() / 2)
        self.static_model_dir = os.path.join(model_dir, 'static')
        self._static_model_file = os.path.join(self.static_model_dir, "inference.pdmodel")
        self._static_params_file = os.path.join(self.static_model_dir, "inference.pdiparams")
        if not os.path.exists(self._static_params_file):
            # paddle动态图模型需要导出为静态图模型，才能预测部署
            logger.info("Converting to the inference model cost a little time.")
            logger.info(f'Loading model from {model_dir}')
            model = UIE.from_pretrained(model_dir)
            model.eval()

            # Convert to static graph with specific input description
            model = paddle.jit.to_static(
                model,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None, None], dtype="int64", name='input_ids'),
                    paddle.static.InputSpec(
                        shape=[None, None], dtype="int64", name='token_type_ids'),
                    paddle.static.InputSpec(
                        shape=[None, None], dtype="int64", name='pos_ids'),
                    paddle.static.InputSpec(
                        shape=[None, None], dtype="int64", name='att_mask'),
                ])
            # Save in static graph model.
            save_path = os.path.join(self.static_model_dir, "inference")
            paddle.jit.save(model, save_path)
            logger.info("The inference model saved:{}".format(self.static_model_dir))

        # Default to use Paddle Inference
        self._predictor_type = 'paddle-inference'
        if self._predictor_type == "paddle-inference":
            self._config = paddle.inference.Config(self._static_model_file,
                                                   self._static_params_file)
            self._prepare_static_mode()
        else:
            self._prepare_onnx_mode()
        logger.debug(f"Use device: {device}")
        logger.debug("Model Loaded.")

    def _prepare_static_mode(self):
        """
        Construct the input data and predictor in the PaddlePaddle static mode.
        """
        if paddle.get_device() == 'cpu':
            self._config.disable_gpu()
            self._config.enable_mkldnn()
        else:
            self._config.enable_use_gpu(100, 0)
            self._config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
        self._config.set_cpu_math_library_num_threads(self._num_threads)
        self._config.switch_use_feed_fetch_ops(False)
        self._config.disable_glog_info()
        self._config.enable_memory_optim()
        self.predictor = paddle.inference.create_predictor(self._config)
        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_handle = [
            self.predictor.get_output_handle(name)
            for name in self.predictor.get_output_names()
        ]

    def _prepare_onnx_mode(self):
        import onnx
        import onnxruntime as ort
        import paddle2onnx
        from onnxconverter_common import float16
        onnx_dir = self.static_model_dir
        if not os.path.exists(onnx_dir):
            os.mkdir(onnx_dir)
        float_onnx_file = os.path.join(onnx_dir, 'model.onnx')
        if not os.path.exists(float_onnx_file):
            onnx_model = paddle2onnx.command.c_paddle_to_onnx(
                model_file=self._static_model_file,
                params_file=self._static_params_file,
                opset_version=13,
                enable_onnx_checker=True)
            with open(float_onnx_file, "wb") as f:
                f.write(onnx_model)
        fp16_model_file = os.path.join(onnx_dir, 'fp16_model.onnx')
        if not os.path.exists(fp16_model_file):
            onnx_model = onnx.load_model(float_onnx_file)
            trans_model = float16.convert_float_to_float16(
                onnx_model, keep_io_types=True)
            onnx.save_model(trans_model, fp16_model_file)
        providers = ['CUDAExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self._num_threads
        sess_options.inter_op_num_threads = self._num_threads
        self.predictor = ort.InferenceSession(
            fp16_model_file, sess_options=sess_options, providers=providers)
        assert 'CUDAExecutionProvider' in self.predictor.get_providers(), \
            f"The environment for GPU inference is not set properly. " \
            "A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. " \
            "Please run the following commands to reinstall: \n " \
            "1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu"


class UIEPredictor(object):
    def __init__(self, model_name_or_path='uie-base', schema=(), max_seq_len=512, position_prob=0.5, device='cpu'):
        if device not in ['cpu', 'gpu']:
            raise ValueError(f"The device must be cpu or gpu, device error: {device}")

        if os.path.exists(os.path.join(model_name_or_path, 'model_state.pdparams')):
            model_dir = model_name_or_path
        else:
            model_name = model_name_or_path
            resource_file_urls = MODEL_MAP[model_name]['resource_file_urls']
            model_dir = os.path.join(USER_DATA_DIR, model_name)
            for key, val in resource_file_urls.items():
                file_path = os.path.join(model_dir, key)
                if not os.path.exists(file_path):
                    logger.info(f"Downloading resource files to {file_path}")
                    get_path_from_url(val, model_dir)
        logger.debug(f"Model dir: {model_dir}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._position_prob = position_prob
        self._max_seq_len = max_seq_len
        self.schema_tree = None
        self.set_schema(schema)
        self.inference_backend = InferBackend(model_dir, device=device)

    def set_schema(self, schema):
        if isinstance(schema, dict) or isinstance(schema, str):
            schema = [schema]
        self.schema_tree = self._build_tree(schema)

    @classmethod
    def _build_tree(cls, schema, name='root'):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError(
                            "Invalid schema, value for each key:value pairs should be list or string"
                            "but {} received".format(type(v)))
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError(
                    "Invalid schema, element should be string or dict, "
                    "but {} received".format(type(s)))
        return schema_tree

    def _single_stage_predict(self, inputs):
        input_texts = []
        prompts = []
        for i in range(len(inputs)):
            input_texts.append(inputs[i]["text"])
            prompts.append(inputs[i]["prompt"])
        # max predict length should exclude the length of prompt and summary tokens
        max_predict_len = self._max_seq_len - len(max(prompts)) - 3

        short_input_texts, self.input_mapping = self._auto_splitter(
            input_texts, max_predict_len, split_sentence=False)

        short_texts_prompts = []
        for k, v in self.input_mapping.items():
            short_texts_prompts.extend([prompts[k] for i in range(len(v))])
        short_inputs = [{
            "text": short_input_texts[i],
            "prompt": short_texts_prompts[i]
        } for i in range(len(short_input_texts))]

        def read(inputs):
            for example in inputs:
                encoded_inputs = self._tokenizer(
                    text=[example["prompt"]],
                    text_pair=[example["text"]],
                    stride=len(example["prompt"]),
                    truncation=True,
                    max_seq_len=self._max_seq_len,
                    pad_to_max_seq_len=True,
                    return_attention_mask=True,
                    return_position_ids=True,
                    return_dict=False)
                encoded_inputs = encoded_inputs[0]

                tokenized_output = [
                    encoded_inputs["input_ids"],
                    encoded_inputs["token_type_ids"],
                    encoded_inputs["position_ids"],
                    encoded_inputs["attention_mask"],
                    encoded_inputs["offset_mapping"]
                ]
                tokenized_output = [
                    np.array(
                        x, dtype="int64") for x in tokenized_output
                ]

                yield tuple(tokenized_output)

        infer_ds = load_dataset(read, inputs=short_inputs, lazy=False)
        batch_sampler = paddle.io.BatchSampler(
            dataset=infer_ds, batch_size=64, shuffle=False)

        infer_data_loader = paddle.io.DataLoader(
            dataset=infer_ds, batch_sampler=batch_sampler, return_list=True)

        sentence_ids = []
        probs = []
        for batch in infer_data_loader:
            input_ids, token_type_ids, pos_ids, att_mask, offset_maps = batch
            if self.inference_backend._predictor_type == "paddle-inference":
                self.inference_backend.input_handles[0].copy_from_cpu(input_ids.numpy())
                self.inference_backend.input_handles[1].copy_from_cpu(token_type_ids.numpy())
                self.inference_backend.input_handles[2].copy_from_cpu(pos_ids.numpy())
                self.inference_backend.input_handles[3].copy_from_cpu(att_mask.numpy())
                self.inference_backend.predictor.run()
                start_prob = self.inference_backend.output_handle[0].copy_to_cpu().tolist()
                end_prob = self.inference_backend.output_handle[1].copy_to_cpu().tolist()
            else:
                input_dict = {
                    "input_ids": input_ids.numpy(),
                    "token_type_ids": token_type_ids.numpy(),
                    "pos_ids": pos_ids.numpy(),
                    "att_mask": att_mask.numpy()
                }
                start_prob, end_prob = self.inference_backend.predictor.run(None, input_dict)

            start_ids_list = get_bool_ids_greater_than(
                start_prob, limit=self._position_prob, return_prob=True)
            end_ids_list = get_bool_ids_greater_than(
                end_prob, limit=self._position_prob, return_prob=True)

            for start_ids, end_ids, ids, offset_map in zip(
                    start_ids_list, end_ids_list,
                    input_ids.tolist(), offset_maps.tolist()):
                for i in reversed(range(len(ids))):
                    if ids[i] != 0:
                        ids = ids[:i]
                        break
                span_list = get_span(start_ids, end_ids, with_prob=True)
                sentence_id, prob = get_id_and_prob(span_list, offset_map)
                sentence_ids.append(sentence_id)
                probs.append(prob)
        results = self._convert_ids_to_results(short_inputs, sentence_ids, probs)
        results = self._auto_joiner(results, short_input_texts, self.input_mapping)
        return results

    def _auto_splitter(self, input_texts, max_text_len, split_sentence=False):
        """
        Split the raw texts automatically for model inference.
        Args:
            input_texts (List[str]): input raw texts.
            max_text_len (int): cutting length.
            split_sentence (bool): If True, sentence-level split will be performed.
        return:
            short_input_texts (List[str]): the short input texts for model inference.
            input_mapping (dict): mapping between raw text and short input texts.
        """
        input_mapping = {}
        short_input_texts = []
        cnt_org = 0
        cnt_short = 0
        for text in input_texts:
            if not split_sentence:
                sens = [text]
            else:
                sens = cut_chinese_sent(text)
            for sen in sens:
                lens = len(sen)
                if lens <= max_text_len:
                    short_input_texts.append(sen)
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = [cnt_short]
                    else:
                        input_mapping[cnt_org].append(cnt_short)
                    cnt_short += 1
                else:
                    temp_text_list = [
                        sen[i:i + max_text_len]
                        for i in range(0, lens, max_text_len)
                    ]
                    short_input_texts.extend(temp_text_list)
                    short_idx = cnt_short
                    cnt_short += math.ceil(lens / max_text_len)
                    temp_text_id = [
                        short_idx + i for i in range(cnt_short - short_idx)
                    ]
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = temp_text_id
                    else:
                        input_mapping[cnt_org].extend(temp_text_id)
            cnt_org += 1
        return short_input_texts, input_mapping

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if short_result == []:
                continue
            elif 'start' not in short_result[0].keys(
            ) and 'end' not in short_result[0].keys():
                is_cls_task = True
                break
            else:
                break
        for k, vs in input_mapping.items():
            if is_cls_task:
                cls_options = {}
                single_results = []
                for v in vs:
                    if len(short_results[v]) == 0:
                        continue
                    if short_results[v][0]['text'] not in cls_options.keys():
                        cls_options[short_results[v][0][
                            'text']] = [1, short_results[v][0]['probability']]
                    else:
                        cls_options[short_results[v][0]['text']][0] += 1
                        cls_options[short_results[v][0]['text']][
                            1] += short_results[v][0]['probability']
                if len(cls_options) != 0:
                    cls_res, cls_info = max(cls_options.items(),
                                            key=lambda x: x[1])
                    concat_results.append([{
                        'text': cls_res,
                        'probability': cls_info[1] / cls_info[0]
                    }])
                else:
                    concat_results.append([])
            else:
                offset = 0
                single_results = []
                for v in vs:
                    if v == 0:
                        single_results = short_results[v]
                        offset += len(short_inputs[v])
                    else:
                        for i in range(len(short_results[v])):
                            if 'start' not in short_results[v][i] or 'end' not in short_results[v][i]:
                                continue
                            short_results[v][i]['start'] += offset
                            short_results[v][i]['end'] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        """
        Convert ids to raw text in a single stage.
        """
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example["text"]
            prompt = example["prompt"]
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 <= end:
                    continue
                if end < 0:
                    start += (len(prompt) + 1)
                    end += (len(prompt) + 1)
                    result = {"text": prompt[start:end], "probability": prob[i]}
                    result_list.append(result)
                else:
                    result = {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "probability": prob[i]
                    }
                    result_list.append(result)
            results.append(result_list)
        return results

    def _multi_stage_predict(self, data):
        """
        Traversal the schema tree and do multi-stage prediction.
        Args:
            data (list): a list of strings
        Returns:
            list: a list of predictions, where the list's length
                equals to the length of `data`
        """
        results = [{} for _ in range(len(data))]
        # input check to early return
        if len(data) < 1 or self.schema_tree is None:
            return results

        # copy to stay `self.schema_tree` unchanged
        schema_list = self.schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                for one_data in data:
                    examples.append({
                        "text": one_data,
                        "prompt": dbc2sbc(node.name)
                    })
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, one_data in zip(node.prefix, data):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            examples.append({
                                "text": one_data,
                                "prompt": dbc2sbc(p + node.name)
                            })
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1
            if len(examples) == 0:
                result_list = []
            else:
                result_list = self._single_stage_predict(examples)

            if not node.parent_relations:
                relations = [[] for i in range(len(data))]
                for k, v in input_map.items():
                    for idx in v:
                        if len(result_list[idx]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[idx]
                        else:
                            results[k][node.name].extend(result_list[idx])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if "relations" not in relations[k][i].keys():
                            relations[k][i]["relations"] = {
                                node.name: result_list[v[i]]
                            }
                        elif node.name not in relations[k][i]["relations"].keys(
                        ):
                            relations[k][i]["relations"][
                                node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relations"][node.name].extend(
                                result_list[v[i]])
                new_relations = [[] for i in range(len(data))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relations" in relations[i][j].keys(
                        ) and node.name in relations[i][j]["relations"].keys():
                            for k in range(
                                    len(relations[i][j]["relations"][
                                            node.name])):
                                new_relations[i].append(relations[i][j][
                                                            "relations"][node.name][k])
                relations = new_relations

            prefix = [[] for _ in range(len(data))]
            for k, v in input_map.items():
                for idx in v:
                    for i in range(len(result_list[idx])):
                        prefix[k].append(result_list[idx][i]["text"] + "的")

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)
        return results

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if isinstance(inputs, str):
            if len(inputs) == 0:
                raise ValueError(
                    "Invalid inputs, input text should not be empty text, please check your input.".
                        format(type(inputs)))
            inputs = [inputs]
        elif isinstance(inputs, list):
            if not (isinstance(inputs[0], str) and len(inputs[0].strip()) > 0):
                raise TypeError(
                    "Invalid inputs, input text should be list of str, and first element of list should not be empty text.".
                        format(type(inputs[0])))
        else:
            raise TypeError(
                "Invalid inputs, input text should be str or list of str, but type of {} found!".
                    format(type(inputs)))
        return inputs

    def predict(self, input_data):
        input_data = self._check_input_text(input_data)
        results = self._multi_stage_predict(input_data)
        return results


class SchemaTree(object):
    """
    Implementation of SchemaTree
    """

    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        self.prefix = None
        self.parent_relations = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(
            node, SchemaTree
        ), "The children of a node should be an instance of SchemaTree."
        self.children.append(node)

    def has_child(self):
        return len(self.children) > 0


def dbc2sbc(s):
    rs = ""
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xfee0
        if not (0x0021 <= code and code <= 0x7e):
            rs += char
            continue
        rs += chr(code)
    return rs


def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely, reference to 
    "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    para = re.sub(r'([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def get_id_and_prob(spans, offset_map):
    prompt_length = 0
    for i in range(1, len(offset_map)):
        if offset_map[i] != [0, 0]:
            prompt_length += 1
        else:
            break

    for i in range(1, prompt_length + 1):
        offset_map[i][0] -= (prompt_length + 1)
        offset_map[i][1] -= (prompt_length + 1)

    sentence_id = []
    prob = []
    for start, end in spans:
        prob.append(start[1] * end[1])
        sentence_id.append((offset_map[start[0]][0], offset_map[end[0]][1]))
    return sentence_id, prob
