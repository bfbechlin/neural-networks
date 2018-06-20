import copy
from utils import encode_matrix_list
import numpy as np
from neural_network import NeuralNetwork

def columnVector(array):
    return np.vstack(np.array(array))

class NeuralNetworkValidation:
    def __init__(self, network, thetas, LAMBDA, dataset):
        self.LAMBDA = LAMBDA
        self.network = network
        self.thetas = thetas
        self.dataset = [(columnVector(inputs), columnVector(outputs)) for (inputs, outputs) in dataset]
        
    def backprogagation(self):
        net = NeuralNetwork(self.network, self.thetas, LAMBDA=self.LAMBDA, ALPHA=0)
        net.trainTurn(self.dataset)
        return [layer.D for layer in net.layers]

    def numericGrad(self, EPSILON=0.0000010000):
        result = copy.deepcopy(self.thetas)
        for layer in range(len(result)):
            for i in range(len(result[layer])):
                for j in range(len(result[layer][i])):
                    thetas_plus = copy.deepcopy(self.thetas)
                    thetas_plus[layer][i][j] += EPSILON

                    thetas_minus = copy.deepcopy(self.thetas)
                    thetas_minus[layer][i][j] -= EPSILON

                    net_plus = NeuralNetwork(self.network, thetas_plus, LAMBDA=self.LAMBDA, ALPHA=0)
                    net_minus = NeuralNetwork(self.network, thetas_minus, LAMBDA=self.LAMBDA, ALPHA=0)

                    net_plus.trainTurn(self.dataset, True)
                    net_minus.trainTurn(self.dataset, True)

                    result[layer][i][j] = (net_plus.computeCost(len(self.dataset)) - net_minus.computeCost(len(self.dataset))) / (2 * EPSILON)

        return result

if __name__ == '__main__':
    network = [1, 2, 1]
    thetas = [
        [
            [0.40000, 0.10000],
	        [0.30000, 0.20000]
        ],
        [
            [0.70000, 0.50000, 0.60000]
        ]
    ]
    dataset = [
        [
            [0.13000],
            [0.90000]
        ],
        [
            [0.42000],
            [0.23000]
        ]
    ]
    
    net = NeuralNetworkValidation(network, thetas, 0, dataset)
    print('#######Example1##########')
    print('-backpropagation:')
    print(encode_matrix_list(net.backprogagation()))
    print('-numeric gradiant:')
    print(encode_matrix_list(net.numericGrad()))

    network = [2, 4, 3, 2]
    thetas = [
        [
            [0.42000, 0.15000, 0.40000],
            [0.72000, 0.10000, 0.54000],
            [0.01000, 0.19000, 0.42000],  
            [0.30000, 0.35000, 0.68000]
        ],
        [
            [0.21000, 0.67000, 0.14000, 0.96000, 0.87000],
            [0.87000, 0.42000, 0.20000, 0.32000, 0.89000],
            [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]
        ],
        [
            [0.04000, 0.87000, 0.42000, 0.53000],
            [0.17000, 0.10000, 0.95000, 0.69000]
        ]
    ]
    dataset = [
        [
            [0.32000, 0.68000],
            [0.75000, 0.98000],
        ],
        [
            [0.83000, 0.02000],
            [0.75000, 0.28000]
        ]
    ]
    net = NeuralNetworkValidation(network, thetas, 0.250, dataset)
    print('#######Example2##########')
    print('-backpropagation:')
    print(encode_matrix_list(net.backprogagation()))
    print('-numeric gradiant:')
    print(encode_matrix_list(net.numericGrad()))