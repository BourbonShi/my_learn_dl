# encoding=utf-8

from node import Node
from constNode import ConstNode


class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes=[]
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)
