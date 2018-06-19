from unittest import TestCase, main
from neural_network import NeuralNetwork
from numpy import array, vstack
from numpy.testing import assert_array_almost_equal
from data import Datapoint

network = [2, 4, 3, 2]
thetas = (
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
)
examples = (
    [0.32000, 0.68000],
    [0.83000, 0.02000]
)
outputs = (
    [0.75000, 0.98000],
    [0.75000, 0.28000]
)
predicteds = (
    [0.83318, 0.84132],
    [0.82953, 0.83832]
)
grads = (
    [
        [0.00804, 0.02564, 0.04987],
        [0.00666, 0.01837, 0.06719],
        [0.00973, 0.03196, 0.05252],
        [0.00776, 0.05037, 0.08492]
    ],
    [
        [0.01071, 0.09068, 0.02512, 0.12597, 0.11586],
        [0.02442, 0.06780, 0.04164, 0.05308, 0.12677],
        [0.03056, 0.08924, 0.12094, 0.10270, 0.03078]
    ],
    [
        [0.08135, 0.17935, 0.12476, 0.13186],
        [0.20982, 0.19195, 0.30343, 0.25249]
    ]
)

class NeuralLayerTest(TestCase):
    def setUp(self):
        self.network = NeuralNetwork(network, thetas, K=3, LAMBDA=0.250)

    def test_correctLayers(self):
        for i, layer in enumerate(self.network.layers):
            self.assertEqual(layer.inputs, network[i])
            self.assertEqual(layer.outputs, network[i+1])
            assert_array_almost_equal(layer.thetas, array(thetas[i]), decimal=5)

    def test_labelToOutputs(self):
        label0 = [
            [1.0],
            [0]
        ]
        label1 = [
            [0],
            [1.0]
        ]
        assert_array_almost_equal(self.network._labelToOutputs(0), array(label0), decimal=5)
        assert_array_almost_equal(self.network._labelToOutputs(1), array(label1), decimal=5)

    def test_outputsToLabel(self):
        label0 = [
            [1.0],
            [0]
        ]
        label1 = [
            [0],
            [1.0]
        ]
        self.assertEqual(self.network._outputsToLabel(label0), 0)
        self.assertEqual(self.network._outputsToLabel(label1), 1)
    
    def test_atributesToInputs(self):
        attr1 = [1.0, 0.56, 0]
        assert_array_almost_equal(self.network._atributesToInputs(attr1), array([[1.0], [0.56], [0]]), decimal=5)

    def test_batchGroups(self):
        dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(self.network._batchGroups(dataset), [[1,2,3], [4,5,6], [7,8,9], [10]])

    def test_forwardPropagation(self):
        for example, predicted in zip(examples, predicteds):
            ex = self.network._atributesToInputs(example)
            pred = self.network._atributesToInputs(predicted)
            assert_array_almost_equal(self.network._forwardPropagation(ex), pred, decimal=5)

    def test_backPropagation(self):
        for example, output in zip(examples, outputs):
            ex = self.network._atributesToInputs(example)
            out = self.network._atributesToInputs(output)
            pred = self.network._forwardPropagation(ex)
            self.network._backPropagation(pred - out)
        
        for i, layer in enumerate(self.network.layers):
            assert_array_almost_equal(layer.updateThetas(len(examples), 0, self.network.LAMBDA), 
                array(grads[i]), decimal=5)

if __name__ == '__main__':
  main()