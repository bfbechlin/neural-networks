from neural_layer import NeuralLayer

class NeuralNetwork:
    def __init__(self, network, classifierFunction = lambda x: x, k=0):
        self.layers = (NeuralLayer(network[i], network[i+1]) for i in range(0, len(network)-1))
        self.classifierFunction = classifierFunction
        self.k = k

    def _batchGroups(self, dataset):
        if self.k == 0:
            return dataset
        else:
            return [dataset[i:i + self.k] for i in range(0, len(dataset), self.k)]

    def train(self, dataset):
        for batch in self._batchGroups(dataset):
            for datapoint in batch:
                pass
    
    def classify(self, datapoint):
        activations = datapoint
        for layer in self.layers:
            activations = layer.computeActivations(activations)
        return self.classifierFunction(activations)