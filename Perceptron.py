# encoding=utf-8

from functools import reduce

class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % ([w for w in self.weights], self.bias)

    def get_map_value(self, map):
        return [m for m in map]

    def predict(self, input_vec):
        return self.activator(
            reduce(lambda a, b: a + b,
                   self.get_map_value(map(lambda x_w: x_w[0] * x_w[1],
                                          zip(input_vec, self.weights))),
                   0.0) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_wieghts(input_vec, output, label, rate)

    def _update_wieghts(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = self.get_map_value(map(lambda x_w: x_w[1] + rate * delta * x_w[0],
                                              zip(input_vec, self.weights)))
        self.bias += rate * delta

