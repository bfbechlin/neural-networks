import numpy as np
import math

def g(x):
    return 1 / (1 + math.exp(-x))

G = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
POW = np.vectorize(lambda x: x**2)

def _removeBias(matrix):
        cp = np.array(matrix)
        cp[:, 0] =  0
        return cp

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
        
    def reset(self):
        self.a = np.zeros((self.outputs, 1))
        self.delta = np.zeros((self.outputs, 1))
        self.D = np.zeros((self.outputs, 1))
        self.n = 0

    '''
        Compute the activations
        Must not contains bias on input
    '''
    def computeActivations(self, inputs):
        x = np.vstack([1., inputs])
        z = np.dot(self.thetas, x)
        self.a = G(z)
        self.n += 1
        return self.a 

    def computeRegularization(self, includeBias=False):
        factors = POW(self.a)
        if not includeBias:
            factors = _removeBias(factors)
        return factors.sum()

    def computeDelta(self, deltaNL, isLast=False):
        if isLast:
            self.delta = self.a - deltaNL
        else:
            aux = np.dot(np.transpose(self.thetas), deltaNL)
            aux = np.dot(aux, self.a)
            self.delta = np.dot(aux, (1 - self.a))
        self.D = self.D + np.dot(self.delta, np.transpose(self.a))
        return self.delta

    def updateThetas(self, ALFA, LAMBDA):
        try:
            D = 1 / self.n * (self.D + LAMBDA * _removeBias(self.thetas))
        except:
            D = 0
        self.thetas = self.thetas - ALFA * D

if __name__ == '__main__':
    a = NeuralLayer(thetas=np.array([[0.4, 0.1], [0.3, 0.2]]))