from copy import copy
import numpy as np
import math

G = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))

class NeuralLayer:
    '''
        inputs: number of inputs neurons, does not count bias neuron
        outputs: number of outputs neurons
        thetas: initial paramaters of each feature
    '''
    def __init__(self, inputs=1, outputs=1, thetas=None):
        self.thetas = (np.random.rand(outputs, inputs + 1) * 2 - 1) if thetas is None else np.array(thetas)
        self.reset()
        
    @property
    def thetasNoBias(self):
        cp = np.array(self.thetas)
        cp[:, 0] = 0
        return cp

    def reset(self):
        self.a = 0
        self.D = 0
        self.J = 0

    def computeActivations(self, inputs):
        self.a = np.vstack((1., inputs)) 
        z = np.dot(self.thetas, self.a)
        return G(z)

    def computeRegularization(self, includeBias=False):
        factors = self.thetas if includeBias else self.thetasNoBias
        return np.sum(factors ** 2)

    def computeDeltas(self, deltaNL):
        confidence = np.multiply(self.a, (1 - self.a))
        delta = np.multiply(np.dot(np.transpose(self.thetas), deltaNL), confidence)
        return np.delete(delta, (0), axis=0)

    def computeAndUpdateGrads(self, deltaNL):
        D = np.dot(deltaNL, np.transpose(self.a))
        self.D = self.D + D
        return D

    def updateThetas(self, n, ALPHA, LAMBDA):
        D = 1.0 / n * (self.D + LAMBDA * self.thetasNoBias)
        self.thetas = self.thetas - ALPHA * D
        return D

    def computeCost(self, n):
        J = float(self.J) / n
        regAcc = 0
        for layer in self.layers:
            regAcc += layer.computeRegularization()
        S = float(self.LAMBDA) / (2.0 *  n) * regAcc
        return J + S

if __name__ == '__main__':
    a = NeuralLayer(thetas=np.array([[0.4, 0.1], [0.3, 0.2]]))