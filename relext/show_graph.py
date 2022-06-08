# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from loguru import logger


class ShowGraph:
    """创建展示页面"""

    def __init__(self):
        """init html"""
        self.base = """
<html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <script type="text/javascript" src="../static/VIS/dist/vis.js"></script>
  <link href="../static/VIS/dist/vis.css" rel="stylesheet" type="text/css">
</head>
<body>

<div id="VIS_draw"></div>

<script type="text/javascript">
  var nodes = data_nodes;
  var edges = data_edges;

  var container = document.getElementById("VIS_draw");

  var data = {
    nodes: nodes,
    edges: edges
  };

  var options = {
      nodes: {
          shape: 'circle',
          size: 15,
          font: {
              size: 15
          }
      },
      edges: {
          font: {
              size: 10,
              align: 'center'
          },
          color: 'red',
          arrows: {
              to: {enabled: true, scaleFactor: 1.2}
          },
          smooth: {enabled: true}
      },
      physics: {
          enabled: true
      }
  };

  var network = new vis.Network(container, data, options);

</script>
</body>
</html>
    """

    def create_html(self, data_nodes, data_edges, html_path="graph_show.html"):
        """
        生成html文件
        :param data_nodes:
        :param data_edges:
        :param html_path:
        :return:
        """
        with open(html_path, 'w', encoding='utf-8') as f:
            html = self.base.replace('data_nodes', str(data_nodes)).replace('data_edges', str(data_edges))
            f.write(html)
            logger.info("Save graph html: {}".format(html_path))
