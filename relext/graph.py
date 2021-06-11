# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 三元组构建图的节点和关系边
"""
from relext.show_graph import ShowGraph


class Graph:
    def __init__(self, triple_list):
        self.triple_list = triple_list
        nodes, edges = self.create_edge()
        self.nodes = nodes
        self.edges = edges

    def create_edge(self):
        nodes = []
        for t in self.triple_list:
            assert len(t) == 3
            nodes.append(t[0])
            nodes.append(t[2])
        node_dict = {node: index for index, node in enumerate(nodes)}

        data_nodes = []
        data_edges = []
        for node, index in node_dict.items():
            data = {"group": 'Event', "id": index, "label": node}
            data_nodes.append(data)

        for t in self.triple_list:
            # node1 index, label, node2 index
            data = {'from': node_dict.get(t[0]), 'label': t[1], 'to': node_dict.get(t[2])}
            data_edges.append(data)
        return data_nodes, data_edges

    def show(self, html_path="graph_show.html"):
        s = ShowGraph()
        s.create_html(self.nodes, self.edges, html_path)

    def __repr__(self):
        return "node: {}, \nedge: {}".format(self.nodes, self.edges)
