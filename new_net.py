from copy import copy
import numpy as np

G = np.vectorize(lambda x: 1 / (1 + math.exp(x)))

def toVector(array):
    return np.transpose(np.array([array]))

def addBias(vec):
    return np.vstack((1., vec))

class NeuralNetwork:
    def __init__(self, network, thetas=None, LAMBDA=0, ALPHA=0.001, K=0, STOP=0.001):
        self.network = len(network)
        if thetas is None:
            self.thetas = [(np.random.rand(network[i], network[i+1]) * 2 - 1) for i in range(0, len(network)-1)]
        else:
            self.thetas = [(np.array(net) for net in network]
        self.reset()

        self.K = K
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.STOP = STOP
    
    def reset(self):
        self.a = [0] * (self.network + 1)
        self.D = [0] * self.network
        self.deltas = [0] * self.network
        self.J = 0

    def forwardPropagation(self, inputs):
        self.a[0] = addBias(inputs)
        for i in range(self.network):
            z = np.dot(self.thetas[i], self.a[i])
            self.a[i+1] = addBias(G(z))
    
    def backPropagation(self, delta):
        for i in reversed(range(self.network - 1)):
            
    
    