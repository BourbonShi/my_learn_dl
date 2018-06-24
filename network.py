
from connections import Connections
from layer import Layer
from connection import Connection


class Network(object):
    def __init__(self, layers):
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layer.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodex
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, data_set, rate, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                self.tran_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, lable, sample, rate):
        self.predict(sample)
        self.calc_dalta(lable)
        self.update_weight(rate)

    def calc_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for layer in self.layer[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return [m for m in map(lambda node: node.output, self.layers[-1].nodes[:-1])]

    def dump(self):
        for layer in self.layers:
            layer.dump()
