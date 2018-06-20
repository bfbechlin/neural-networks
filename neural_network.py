import math
import numpy as np
from neural_layer import NeuralLayer

def J_f(y, p):
    return -1.0 * (y * math.log(p) + (1 - y) * math.log(1 - p))

class NeuralNetwork:
    def __init__(self, network, thetas=None, K=0, ALPHA=0.001, LAMBDA=0, STOP=0.001):
        self.inputs = network[0]
        self.outputs = network[-1]
        self.layers = [NeuralLayer(network[i], network[i+1], None if thetas is None else thetas[i]) for i in range(0, len(network)-1)]
        self.J = 0
        self.K = K
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.STOP = STOP

    def _labelToOutputs(self, label):
        #return np.array([[label]])
        out = np.zeros((self.outputs, 1))
        out[label][0] = 1.0
        return out

    def _outputsToLabel(self, output):
        #return 1 if output[0][0] > 0.5 else 0
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
            return [dataset]
        else:
            return [dataset[i:i + self.K] for i in range(0, len(dataset), self.K)]

    def _J(self, outputs, predictions):
        J = np.zeros((self.outputs, 1))
        for i, (out, pred) in enumerate(zip(outputs, predictions)):
            J[i][0] = J_f(out, pred)
        return J

    def forwardPropagation(self, inputs):
        activations = inputs
        print 'inputs', inputs
        for layer in self.layers:
            activations = layer.computeActivations(activations)
            print 'activation', activations
        return activations

    def backPropagation(self, lastDelta):
        deltas = lastDelta
        first = len(self.layers) - 1
        for i, layer in enumerate(reversed(self.layers)):
            layer.computeAndUpdateGrads(deltas)
            if i != first:
                deltas = layer.computeDeltas(deltas)

    def resetLayers(self):
        self.J = 0
        for layer in self.layers:
            layer.reset()

    def updateTethas(self, n):
        gradsAcc = 0
        for layer in self.layers:
            grads = layer.updateThetas(n, self.ALPHA, self.LAMBDA)
            gradsAcc += np.sum(grads ** 2)
        return self.computeCost(n)

    def computeCost(self, n):
        J = float(self.J) / n
        regAcc = 0
        for layer in self.layers:
            regAcc += layer.computeRegularization()
        S = float(self.LAMBDA) / (2.0 *  n) * regAcc
        return J + S

    def trainTurn(self, datapoints, updateCost=False):
        self.resetLayers()
        print 'turn datapoints', datapoints
        for (inputs, outputs) in datapoints:
            predictions = self.forwardPropagation(inputs)
            print 'predictions', predictions
            self.backPropagation(predictions - outputs)
            if updateCost:
                self.J += np.sum(self._J(outputs, predictions))
        return self.updateTethas(len(datapoints))

    def train(self, dataset):
        dataset = [
            (self._atributesToInputs(datapoint.attributes), self._labelToOutputs(datapoint.label))
            for datapoint in dataset
        ]
        err = float('Inf')
        while err > self.STOP:
            self.STOP = float('+inf')
            err = 0
            for batch in self._batchGroups(dataset):
                print self
                err += self.trainTurn(batch, True)
            print(err)

    def classify(self, datapoint):
        inputs = self._atributesToInputs(datapoint.attributes)
        preds = self.forwardPropagation(inputs)
        return self._outputsToLabel(preds)

    def __repr__(self):
        return 'network thetas\n' + '\n'.join([str(l) for l in self.layers])
