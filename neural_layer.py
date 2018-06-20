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
        self.inputs = inputs
        self.outputs = outputs
        self.thetas = np.random.rand(self.outputs, self.inputs + 1) if thetas is None else np.array(thetas)
        self.reset()

    @property
    def thetasNoBias(self):
        cp = np.array(self.thetas)
        cp[:, 0] = 0
        return cp

    def reset(self):
        self.x = 0
        self.a = 0
        self.D = 0

    def computeActivations(self, inputs):
        self.x = np.vstack((1., inputs))
        z = np.dot(self.thetas, self.x)
        self.a = G(z)
        return self.a

    def computeRegularization(self, includeBias=False):
        factors = self.thetas if includeBias else self.thetasNoBias
        return np.sum(factors ** 2)

    def computeDeltas(self, deltaNL):
        confidence = np.multiply(self.x, (1 - self.x))
        delta = np.multiply(np.dot(np.transpose(self.thetas), deltaNL), confidence)
        return np.delete(delta, (0), axis=0)

    def computeAndUpdateGrads(self, deltaNL):
        D = np.dot(deltaNL, np.transpose(self.x))
        self.D = self.D + D
        return D

    def updateThetas(self, n, ALPHA, LAMBDA):
        self.D = 1.0 / n * (self.D + LAMBDA * self.thetasNoBias)
        self.thetas = self.thetas - ALPHA * self.D
        return self.D

    def __repr__(self):
        return str(self.thetas)

if __name__ == '__main__':
    a = NeuralLayer(thetas=np.array([[0.4, 0.1], [0.3, 0.2]]))
