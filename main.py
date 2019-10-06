import random

import numpy as np


def mse(y_set, results):
    avg_values = []
    for y, res in zip(y_set, results):
        values = []
        for y_item, res_item in zip(y, res):
            value = (y_item - res_item) ** 2
            values.append(value)
        avg_values.append(np.average(values))
    return np.average(avg_values)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neurone:
    def __init__(self, weights=None, activation='relu'):
        self.weights = weights
        self.prev_delta_weights = [0 for _ in self.weights]
        self.values = None
        self.output = None
        self.activation = activation

        self.activations = {
            'relu': lambda x: max(0, x),
            'sigmoid': sigmoid
        }

    def __repr__(self):
        return f'<Neurone> weights: {self.weights}'

    def execute(self, values):
        self.values = values
        activation = self.activations[self.activation]
        output = sum([value * weight for value, weight in zip(values, self.weights)])
        self.output = activation(output)

        return self.output


class InputNeurone(Neurone):
    def __init__(self, weights=None, activation=None):
        super().__init__(weights, activation)
        self.activation = 'simple'
        self.activations = {
            'simple': lambda x: x
        }


class Layer:
    def __init__(self, neurons, weights_count=None, input_layer=False, activation=None):
        self.input_layer = input_layer

        if input_layer:
            neuron_type = InputNeurone
            weights = lambda: (1,)
        else:
            neuron_type = Neurone
            weights = lambda: [random.uniform(0, 1) for _ in range(weights_count)]

        self.layer = [neuron_type(weights(), activation) for _ in range(neurons)]

    def calc_layer(self, data):
        result = []
        if self.input_layer:
            for neuron, data_item in zip(self.layer, data):
                result.append(neuron.execute((data_item,)))
        else:
            for neuron in self.layer:
                result.append(neuron.execute(data))

        return result

    def __repr__(self):
        return f'<Layer> neurons: {self.layer}'


class Network:
    def __init__(self):
        self.network = []

        self.loss = {
            'mse': mse
        }

    def calculate_error(self, results, y_set, loss):
        return self.loss[loss](y_set, results)

    def add_layer(self, neurones, input_dim=None, activation=None):
        if not self.network:
            self.network.append(Layer(input_dim, input_layer=True))
        self.network.append(Layer(neurones, weights_count=len(self.network[-1].layer), activation=activation))

    def calc_network(self, x_set):
        data = x_set
        for layer in self.network:
            data = layer.calc_layer(data)
        return data

    def train(self, expected, output):
        expected = expected
        output = output
        deltas = []
        for layer_index, layer in enumerate(reversed(self.network)):
            tmp_deltas = []
            for neuron_index, neuron in enumerate(layer.layer):
                if layer_index == 0:
                    delta_neuron = (expected[neuron_index] - output[neuron_index]) * (1 - output[neuron_index]) * output[neuron_index]
                    tmp_deltas.append(delta_neuron)
                else:
                    prev_layer = self.network[-layer_index].layer
                    delta_neuron = ((1 - neuron.output) * neuron.output) * sum(
                        (prev_layer_neuron.weights[neuron_index] * delta for prev_layer_neuron, delta in
                         zip(prev_layer, deltas))
                    )
                    delta_neuron = delta_neuron
                    tmp_deltas.append(delta_neuron)
                    for prev_layer_neuron, delta in zip(prev_layer, deltas):
                        grad = neuron.output * delta
                        delta_weight = 0.7 * grad + prev_layer_neuron.prev_delta_weights[neuron_index] * 0.3
                        prev_layer_neuron.prev_delta_weights[neuron_index] = delta_weight
                        prev_layer_neuron.weights[neuron_index] += delta_weight
            deltas = tmp_deltas[:]
            tmp_deltas.clear()

    def fit(self, x_set, y_set, epochs=1, loss='mse'):
        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}/{epochs}')
            results = []
            for x, y in zip(x_set, y_set):
                output = self.calc_network(x)
                self.train(expected=y, output=output)
                results.append(output)
            error = self.calculate_error(results, y_set, loss)
            print(f'loss: {error}')

    def predict(self, test_sets):
        return [self.calc_network(test_set) for test_set in test_sets]

    def __repr__(self):
        return str(self.network)


def main():
    network = Network()
    network.add_layer(2, input_dim=2, activation='sigmoid')
    network.add_layer(10, activation='sigmoid')
    network.add_layer(1, activation='sigmoid')

    x_train_set = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train_set = [[0], [1], [1], [0]]

    x_test_set = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_test_set = [[0], [1], [1], [0]]

    network.fit(x_train_set, y_train_set, epochs=50000)

    predictions = network.predict(x_test_set)

    for prediction, y_test in zip(predictions, y_test_set):
        print(f'Expected: {y_test}, output: {prediction}')


if __name__ == '__main__':
    main()
