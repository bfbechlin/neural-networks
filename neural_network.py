from neural_layer import NeuralLayer

class NeuralNetwork:
    def __init__(self, network, thetas=None, dataTransformer=None, K=0, ALPHA=0.001, LAMBDA=0):
        self.inputs = network[0]
        self.outputs = network[-1]
        self.layers = (NeuralLayer(network[i], network[i+1]) for i in range(0, len(network)-1))
        self.K = K
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA

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
        for layer in self.layers:
            layer.updateThetas(n, self.ALPHA, self.LAMBDA)

    def train(self, dataset):
        for batch in self._batchGroups(dataset):
            self._resetLayers()
            for datapoint in batch:
                outputs = self._forwardPropagation(datapoint.inputs)
                self._backPropagation(outputs - datapoint.outputs)
            self._updateTethas(len(batch))               
    
    def classify(self, datapoint):
        return self._forwardPropagation(datapoint.inputs)