import numpy as np
from neural_layer import NeuralLayer

class NeuralNetwork:
    def __init__(self, network, thetas=None, K=0, ALPHA=0.001, LAMBDA=0):
        self.inputs = network[0]
        self.outputs = network[-1]
        self.layers = [NeuralLayer(network[i], network[i+1], None if thetas is None else thetas[i]) for i in range(0, len(network)-1)]
        self.K = K
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA

    def _labelToOutputs(self, label):
        out = np.zeros((self.outputs, 1))
        out[label][0] = 1.0
        return out
    
    def _outputsToLabel(self, output):
        maxValue = 0
        maxIndex = 0
        for i, row in enumerate(output):
            value = row[0]
            if value > maxValue:
                maxIndex = i
                maxValue = value
        return maxIndex

    def _atributesToInputs(self, attributes):
        return np.vstack(np.array(attributes))

    def _batchGroups(self, dataset):
        if self.K == 0:
            return dataset
        else:
            return [dataset[i:i + self.K] for i in range(0, len(dataset), self.K)]

    def _forwardPropagation(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = layer.computeActivations(activations)
        return activations

    def _backPropagation(self, lastDelta):
        deltas = lastDelta
        first = len(self.layers) - 1
        for i, layer in enumerate(reversed(self.layers)):
            layer.computeAndUpdateGrads(deltas)
            if i != first:
                deltas = layer.computeDeltas(deltas)

    def _resetLayers(self):
        for layer in self.layers:
            layer.reset()

    def _updateTethas(self, n):
        gradsAcc = 0
        for layer in self.layers:
            grads = layer.updateThetas(n, self.ALPHA, self.LAMBDA)
            gradsAcc = np.sum(grads ** 2)
        return gradsAcc

    def train(self, dataset):
        for batch in self._batchGroups(dataset):
            self._resetLayers()
            for datapoint in batch:
                outputs = self._forwardPropagation(self._atributesToInputs(datapoint.atributes))
                self._backPropagation(outputs - self._labelToOutputs(datapoint.label))
            self._updateTethas(len(batch))               
    
    def classify(self, datapoint):
        return self._forwardPropagation(datapoint.inputs)