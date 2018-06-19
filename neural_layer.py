import numpy as np
import math

G = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
POW = np.vectorize(lambda x: x**2)

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
        self.x = np.zeros((self.inputs, 1))
        self.a = np.zeros((self.outputs, 1))
        self.D = np.zeros((self.outputs, 1))

    '''
        Compute the activations
        Must not contains bias on input
    '''
    def computeActivations(self, inputs):
        self.x = np.vstack((1., inputs)) 
        z = np.dot(self.thetas, self.x)
        self.a = G(z)
        return self.a 

    def computeRegularization(self, includeBias=False):
        factors = POW(self.a)
        if not includeBias:
            factors = self.thetasNoBias
        return factors.sum()

    def computeDeltas(self, deltaNL):
        confidence = np.multiply(self.x, (1 - self.x))
        delta = np.multiply(np.dot(np.transpose(self.thetas), deltaNL), confidence)
        return np.delete(delta, (0), axis=0)

    def computeAndUpdateGrads(self, deltaNL):
        D = np.dot(deltaNL, np.transpose(self.x))
        self.D = self.D + D
        return D

    def updateThetas(self, n, ALPHA, LAMBDA):
        D = 1.0 / n * (self.D + LAMBDA * self.thetasNoBias)
        self.thetas = self.thetas - ALPHA * D

if __name__ == '__main__':
    a = NeuralLayer(thetas=np.array([[0.4, 0.1], [0.3, 0.2]]))