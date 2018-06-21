from copy import deepcopy, copy
import numpy as np
import math

G = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
def J(y, p):
    return -1.0 * (y * math.log(p) + (1 - y) * math.log(1 - p))

def toVector(array):
    return np.transpose(np.array([array]))

def addBias(vec):
    return np.vstack((1., vec))

def removeBias(vec):
    return np.array(np.delete(vec, (0), axis=0))

def removeBiasMatrix(matrix):
    cp = np.array(matrix)
    cp[:, 0] = 0
    return cp

class NeuralNetwork:
    def __init__(self, network, thetas=None, LAMBDA=0, ALPHA=0.001, K=0, STOP=200, BETA=0):
        self.network = network
        self.inputs = network[0]
        self.outputs = network[-1]
        self.layers = len(network) - 1
        self.sizes = [(network[i+1], network[i] + 1) for i in range(0, len(network) - 1)]
        if thetas is None:
            self.thetas = [(np.random.rand(l, c) * 2 - 1.0) for (l, c) in self.sizes]
        else:
            self.thetas = [np.array(theta) for theta in thetas]
        self.a = [None] * (self.layers + 1)
        self.deltas = [None] * (self.layers + 1)
        self.reset()

        self.K = K
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.STOP = STOP
        self.BETA = BETA

    def reset(self):
        self.D = [np.zeros(size) for size in self.sizes]
        self.zz = [0] * (self.layers + 1)
        self.J = 0.0

    def labelToOutputs(self, label):
        out = np.zeros((self.outputs, 1))
        out[label][0] = 1.0
        return out

    def outputsToLabel(self, outputs):
        maxValue = 0
        maxIndex = 0
        for i, row in enumerate(outputs):
            value = row[0]
            if value > maxValue:
                maxIndex = i
                maxValue = value
        return maxIndex

    def batchGroups(self, dataset):
        if self.K == 0:
            return [dataset]
        else:
            return [dataset[i:i + self.K] for i in range(0, len(dataset), self.K)]

    def forwardPropagation(self, inputs):
        self.a[0] = inputs
        for i in range(self.layers):
            self.a[i] = addBias(self.a[i])
            z = np.dot(self.thetas[i], self.a[i])
            self.a[i+1] = G(z)
        return self.a[-1]

    def backPropagation(self, delta):
        self.deltas[self.layers] = copy(delta)
        for j in reversed(range(1, self.layers + 1)):
            i = j - 1
            confidence = (self.a[i] * (1 - self.a[i]))
            delta = np.multiply(np.dot(np.transpose(self.thetas[i]), self.deltas[j]), confidence)
            self.deltas[i] = np.delete(delta, (0), axis=0)
        for i in range(self.layers):
            self.D[i] = self.D[i] + np.dot(self.deltas[i + 1], np.transpose(self.a[i]))

    def computeCost(self, n):
        regAcc = 0
        for i in range(self.layers):
            regAcc += np.sum(removeBiasMatrix(self.thetas[i]) ** 2)
        S = self.LAMBDA / (2.0 *  n) * regAcc
        return (self.J / n) + S

    def updateCost(self, outputs, predictions):
        for out, pred in zip(outputs, predictions):
            self.J += J(out[0], pred[0])

    def updateGrads(self, n):
        for i in range(self.layers):
            P = self.LAMBDA * removeBiasMatrix(self.thetas[i])
            self.D[i] = (self.D[i] + P) * 1.0 / n

    def updateTethas(self, n):
        for i in range(self.layers):
            self.zz[i] = self.BETA * self.zz[i] + self.D[i]
            self.thetas[i] = self.thetas[i] - self.ALPHA * self.zz[i]

    def trainTurn(self, dataset, updateCost=False):
        self.reset()
        for (inputs, outputs) in dataset:
            predictions = self.forwardPropagation(inputs)
            self.backPropagation(predictions - outputs)
            if updateCost:
                self.updateCost(outputs, predictions)
        self.updateGrads(len(dataset))
        self.updateTethas(len(dataset))
        return self.computeCost(len(dataset))

    def train(self, dataset):
        dataset = [
            [toVector(datapoint.attributes), self.labelToOutputs(datapoint.label)]
            for datapoint in dataset
        ]

        Js = []
        for i in range(self.STOP):
            Js.append([self.trainTurn(batch) for batch in self.batchGroups(dataset)])
        return Js

    def classify(self, datapoint):
        inputs = toVector(datapoint.attributes)
        preds = self.forwardPropagation(inputs)
        return self.outputsToLabel(preds)
