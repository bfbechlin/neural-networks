import numpy as np
import math

def g(x):
    return 1 / (1 + math.exp(-x))

G = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
POW = np.vectorize(lambda x: x**2)

class NeuralLayer:

    '''
        inputs: number of inputs neurons, does not count bias neuron
        outputs: number of outputs neurons
        thetas: initial paramaters of each feature
    '''
    def __init__(self, **kwargs):
        self.inputs = kwargs.get('inputs', 1)
        self.outputs = kwargs.get('outputs', 1)
        self.thetas = kwargs.get('thetas', np.random.rand(self.outputs, self.inputs + 1))
        self.activations = np.zeros((self.outputs, 1))

    '''
        Compute the activations
        Must not contains bias on input
    '''
    def computeActivations(self, inputs):
        x = np.vstack([1., inputs])
        z = np.dot(self.thetas, x)
        self.activations = G(z)
        return self.activations

    def computeRegularization(self, includeBias=False):
        factors = POW(self.activations)
        if not includeBias:
            factors = np.delete(factors, 0, 1)
        return factors.sum()

if __name__ == '__main__':
    a = NeuralLayer(thetas=np.array([[0.4, 0.1], [0.3, 0.2]]))