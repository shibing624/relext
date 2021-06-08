![alt text](docs/public/relext.jpg)

[![PyPI version](https://badge.fury.io/py/relext.svg)](https://badge.fury.io/py/relext)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![Python3](https://img.shields.io/badge/Python-3.X-red.svg)

# relext
RelExt: A Tool for Relation Extraction from Text.

文本关系抽取工具。

**Guide**

- [Question](#Question)
- [Solution](#Solution)
- [Feature](#Feature)
- [Install](#install)
- [Usage](#usage)
- [Dataset](#Dataset)
- [Contact](#Contact)
- [Cite](#Cite)
- [Reference](#reference)

# Question

关系抽取咋做？

# Solution


关系抽取结果为三元组（triple），是一种图数据结构，知识图谱的最小单元，表示两个节点及它们之间的关系，即node1，edge，node2。

语言学上，提取句子主干，如"姚明是李秋平的徒弟"主干为（姚明，徒弟，李秋平），形式化表示为（主语，谓语，宾语），也称为SPO三元组（subject，predicate，object），跟三元组同义。

不同结构化程度的文本，关系抽取(三元组抽取)方法不一样：

- 结构化文本：映射规则即可转化为三元组，相对简单，业务依赖强。
- 非结构化文本：关系抽取包括两个子任务，实体识别，实体关系分类。三元组抽取模型分为以下两类，
	1. pipeline模型：先基于序列标注模型识别文本的实体，再用分类器识别实体间的关系。优点：各模型单独训练，需要训练样本少，适合冷启动；缺点：模型误差传递。
	2. 联合（joint）模型：实体识别模型和实体关系分类模型整合到一个模型，共享底层特征、二者损失值联合训练。优点：误差传递小，模型推理快；缺点：需要大量训练样本。

# Feature


### 开放域文本关系抽取

- GPT2 Model
- Sequence To Sequence Model(seq2seq)
- Taobao dataset


# Install

- Requirements and Installation

The project is based on transformers 4.4.2+, torch 1.6.0+ and Python 3.6+.
Then, simply do:

```
pip3 install relext
```

or

```
git clone https://github.com/shibing624/relext.git
cd relext
python3 setup.py install
```

# Usage
## 问答型对话（Search Bot）

示例[base_demo.py](examples/base_demo.py)

```python
import relext import Bot

bot = Bot()
response = bot.answer('姚明多高呀？')
print(response)
```

output:

```
query: "姚明多高呀？"

answer: "226cm"
```


## 聊天型对话（Generative Bot）

### GPT2模型使用
基于GPT2生成模型训练的聊天型对话模型。

在[模型分享](#模型分享)中下载模型，将模型文件夹model_epoch40_50w下的文件放到自己指定目录`your_model_dir`下：
```
model_epoch40_50w
├── config.json
├── pytorch_model.bin
└── vocab.txt
```

示例[genbot_demo.py](examples/genbot_demo.py)


```python
import relext import Bot

bot = Bot(gpt_model_path=your_model_dir)
response = bot.answer('亲 你吃了吗？', use_gen=True, use_search=False, use_task=False)
print(response)
```

output:

```
query: "亲 吃了吗？"

answer: "吃了"
```


# Dataset

### 关系抽取语料分享
| 关系抽取语料名称 | 数据集地址 |语料描述|
|---------|--------|--------|
|常见中文闲聊|[chinese_chatbot_corpus](https://github.com/codemayq/chinese_chatbot_corpus)|包含小黄鸡语料、豆瓣语料、电视剧对白语料、贴吧论坛回帖语料、微博语料、PTT八卦语料、青云语料等|
|50w中文闲聊语料 | [百度网盘(提取码:4g5e)](https://pan.baidu.com/s/1M87Zf9e8iBqqmfTkKBWBWA) 或 [GoogleDrive](https://drive.google.com/drive/folders/1QFRsftLNTR_D3T55mS_FocPEZI7khdST?usp=sharing) |包含50w个多轮对话的原始语料、预处理数据|


中文关系抽取语料的内容样例如下:
```

```

### 模型分享

|模型 | 共享地址 |模型描述|
|---------|--------|--------|
|model_epoch40_50w | [百度网盘(提取码:aisq)](https://pan.baidu.com/s/11KZ3hU2_a2MtI_StXBUKYw) 或 [GoogleDrive](https://drive.google.com/drive/folders/18TG2sKkHOZz8YlP5t1Qo_NqnGx9ogNay?usp=sharing) |使用50w多轮对话语料训练了40个epoch，loss降到2.0左右。|


# Contact

- 邮件我：xuming: xuming624@qq.com.
- 微信我：
加我*微信号：xuming624, 备注：个人名称-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Cite

如果你在研究中使用了relext，请按如下格式引用：

```latex
@software{relext,
  author = {Xu Ming},
  title = {relext: A Tool for Relation Extraction from Text},
  year = {2021},
  url = {https://github.com/shibing624/relext},
}
```

# License


授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加relext的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python setup.py test`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。


# Reference

- [RelExt paper](./docs/RelExt- A Tool for Relation Extraction from Text in Ontology Extension.pdf)
- [TextGrapher](https://github.com/liuhuanyong/TextGrapher)
