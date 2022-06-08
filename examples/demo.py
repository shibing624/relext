# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
# 实体抽取
from pprint import pprint
from paddlenlp import Taskflow
schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
ie = Taskflow('information_extraction', schema=schema)
pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")) # Better print results using pprint
# >>>
# [{'时间': [{'end': 6, 'probability': 0.9857378532924486, 'start': 0, 'text': '2月8日上午'}],
#   '赛事名称': [{'end': 23,'probability': 0.8503089953268272,'start': 6,'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
#   '选手': [{'end': 31,'probability': 0.8981548639781138,'start': 28,'text': '谷爱凌'}]}]

# 事件抽取
schema = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']} # Define the schema for event extraction
ie.set_schema(schema) # Reset schema
r = ie('中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。')
pprint(r)
# >>>
# [{'地震触发词':
#   [{'end': 58,'probability': 0.9987181623528585,'start': 56,'text': '地震',
#     'relations':
#       {'地震强度': [{'end': 56,'probability': 0.9962985320905915,'start': 52,'text': '3.5级'}],
#       '时间': [{'end': 22,'probability': 0.9882578028575182,'start': 11,'text': '5月16日06时08分'}],
#       '震中位置': [{'end': 50,'probability': 0.8551417444021787,'start': 23,'text': '云南临沧市凤庆县(北纬24.34度，东经99.98度)'}],
#       '震源深度': [{'end': 67,'probability': 0.999158304648045,'start': 63,'text': '10千米'}]}
#     }]
# }]
schema = ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
ie.set_schema(schema)
pprint(ie("（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。"))
# [{'肝癌级别': [{'end': 20,
#             'probability': 0.9243267447402701,
#             'start': 13,
#             'text': 'II-III级'}],
#   '肿瘤的个数': [{'end': 84,
#             'probability': 0.7538413804059623,
#             'start': 82,
#             'text': '1个'}],
#   '肿瘤的大小': [{'end': 100,
#             'probability': 0.8341128043459491,
#             'start': 87,
#             'text': '4.2×4.0×2.8cm'}],
#   '脉管内癌栓分级': [{'end': 70,
#               'probability': 0.9083292325934664,
#               'start': 67,
#               'text': 'M0级'}]}]

# 关系抽取
schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']} # Define the schema for relation extraction
ie.set_schema(schema) # Reset schema
pprint(ie('2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。'))
# [{'竞赛名称': [{'end': 13,
#             'probability': 0.7825402622754041,
#             'relations': {'主办方': [{'end': 22,
#                                   'probability': 0.8421710521379353,
#                                   'start': 14,
#                                   'text': '中国中文信息学会'},
#                                   {'end': 30,
#                                   'probability': 0.7580801847701935,
#                                   'start': 23,
#                                   'text': '中国计算机学会'}],
#                           '已举办次数': [{'end': 82,
#                                     'probability': 0.4671295049136148,
#                                     'start': 80,
#                                     'text': '4届'}],
#                           '承办方': [{'end': 39,
#                                   'probability': 0.8292706618236352,
#                                   'start': 35,
#                                   'text': '百度公司'},
#                                   {'end': 72,
#                                   'probability': 0.6193477885474685,
#                                   'start': 56,
#                                   'text': '中国计算机学会自然语言处理专委会'},
#                                   {'end': 55,
#                                   'probability': 0.7000497331473241,
#                                   'start': 40,
#                                   'text': '中国中文信息学会评测工作委员会'}]},
#             'start': 0,
#             'text': '2022语言与智能技术竞赛'}]}]

# 评论观点抽取，是指抽取文本中包含的评价维度、观点词。
schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']} # Define the schema for opinion extraction
ie.set_schema(schema) # Reset schema
pprint(ie("店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队")) # Better print results using pprint
# [{'评价维度': [{'end': 20,
#             'probability': 0.9817040258681473,
#             'relations': {'情感倾向[正向，负向]': [{'probability': 0.9966142505350533,
#                                           'text': '正向'}],
#                           '观点词': [{'end': 22,
#                                   'probability': 0.957396472711558,
#                                   'start': 21,
#                                   'text': '高'}]},
#             'start': 17,
#             'text': '性价比'},
#           {'end': 2,
#             'probability': 0.9696849569741168,
#             'relations': {'情感倾向[正向，负向]': [{'probability': 0.9982153274927796,
#                                           'text': '正向'}],
#                           '观点词': [{'end': 4,
#                                   'probability': 0.9945318044652538,
#                                   'start': 2,
#                                   'text': '干净'}]},
#             'start': 0,
#             'text': '店面'}]}]

# 情感倾向分类
schema = '情感倾向[正向，负向]' # Define the schema for sentence-level sentiment classification
ie.set_schema(schema) # Reset schema
pprint(ie('这个产品用起来真的很流畅，我非常喜欢'))
# [{'情感倾向[正向，负向]': [{'text': '正向', 'probability': 0.9988661643929895}]}]

# 跨任务抽取, 例如在法律场景同时对文本进行实体抽取和关系抽取，schema可按照如下方式进行构造：
schema = ['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}]
ie.set_schema(schema)
pprint(ie("北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。")) # Better print results using pprint
# [{'原告': [{'end': 37,
#           'probability': 0.9949814024296764,
#           'relations': {'委托代理人': [{'end': 46,
#                                   'probability': 0.7956844697990384,
#                                   'start': 44,
#                                   'text': '李四'}]},
#           'start': 35,
#           'text': '张三'}],
#   '法院': [{'end': 10,
#           'probability': 0.9221074192336651,
#           'start': 0,
#           'text': '北京市海淀区人民法院'}],
#   '被告': [{'end': 67,
#           'probability': 0.8437349536631089,
#           'relations': {'委托代理人': [{'end': 92,
#                                   'probability': 0.7267121388225029,
#                                   'start': 90,
#                                   'text': '赵六'}]},
#           'start': 64,
#           'text': 'B公司'}]}]
