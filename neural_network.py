from copy import copy
import math
import numpy as np
from neural_layer import NeuralLayer

def J_f(y, p):
    return -1.0 * (y * math.log(p) + (1 - y) * math.log(1 - p))

class NeuralNetwork:
    def __init__(self, network, thetas=None, LAMBDA=0, ALPHA=0.001, K=0, STOP=0.001):
        self.inputs = network[0]
        self.outputs = network[-1]
        self.layers = [NeuralLayer(network[i], network[i+1], None if thetas is None else thetas[i]) for i in range(0, len(network)-1)]
        for i, layer in enumerate(self.layers):
            print()
            print(layer.thetas)
        
        self.J = 0
        self.K = K
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.STOP = STOP

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
        return np.transpose(np.array([attributes]))

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
        for layer in self.layers:
            activations = layer.computeActivations(activations)
        return activations

    def backPropagation(self, lastDelta):
        deltas = copy(lastDelta)
        first = len(self.layers) - 1
        for layer in reversed(self.layers):
            layer.computeAndUpdateGrads(deltas)
            deltas = copy(layer.computeDeltas(deltas))

    def resetLayers(self):
        self.J = 0
        for layer in self.layers:
            layer.reset()

    def updateTethas(self, n):
        gradsAcc = 0
        for layer in self.layers:
            grads = layer.updateThetas(n, self.ALPHA, self.LAMBDA)
            gradsAcc += np.sum(grads ** 2)
        return gradsAcc

    def computeCost(self, n):
        J = float(self.J) / n
        regAcc = 0
        for layer in self.layers:
            regAcc += layer.computeRegularization()
        S = float(self.LAMBDA) / (2.0 *  n) * regAcc
        return J + S

    def trainTurn(self, datapoints, updateCost=False):
        self.resetLayers()
        for (inputs, outputs) in datapoints:
            predictions = self.forwardPropagation(inputs)
            print('A')
            #print(outputs)
            print(predictions)
            print(predictions - outputs)
            self.backPropagation(predictions - outputs)
            #print(self.layers[-1].D)
            if updateCost:
                self.J += np.sum(self._J(outputs, predictions))
        a = self.updateTethas(len(datapoints))
        #print(self.layers[-1].D)
        return a

    def train(self, dataset):
        dataset = [
            [self._atributesToInputs(datapoint.attributes), self._labelToOutputs(datapoint.label)]
            for datapoint in dataset
        ]
        err = float('Inf')
        for i in range(1):
            err = 0
            for batch in self._batchGroups(dataset):
                err += self.trainTurn(batch)
            print(err)

        for layer in self.layers:
            print(layer.inputs)
            print(layer.outputs)
            print(layer.thetas)
    
    def classify(self, datapoint):
        self.resetLayers()
        inputs = self._atributesToInputs(datapoint.attributes)
        preds = self.forwardPropagation(inputs)
        return self._outputsToLabel(preds)

