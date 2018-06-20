from unittest import TestCase, main
from new_net import NeuralNetwork, toVector
import numpy as np
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
J = (
    [0.791],
    [1.944]
)
Jtotal = 1.90351
grads = (
    (
        [
            [-0.00087, -0.00028, -0.00059],
			[-0.00133, -0.00043, -0.00091],
			[-0.00053, -0.00017, -0.00036],
			[-0.00070, -0.00022, -0.00048]
        ],
        [
            [0.00639, 0.00433, 0.00482, 0.00376, 0.00451],
			[-0.00925, -0.00626, -0.00698, -0.00544, -0.00653],
			[-0.00779, -0.00527, -0.00587, -0.00458, -0.00550],
        ],
        [
            [0.08318, 0.07280, 0.07427, 0.06777],
			[-0.13868, -0.12138, -0.12384, -0.11300]
        ]
    ),
    (
        [
            [0.01694, 0.01406, 0.00034],
			[0.01465, 0.01216, 0.00029],
			[0.01999, 0.01659, 0.00040],
			[0.01622, 0.01346, 0.00032]
        ],
        [
            [0.01503, 0.00954, 0.01042, 0.00818, 0.00972],
			[0.05809, 0.03687, 0.04025, 0.03160, 0.03756],
			[0.06892, 0.04374, 0.04775, 0.03748, 0.04456]
        ],
        [
            [0.07953, 0.06841, 0.07025, 0.06346],
			[0.55832, 0.48027, 0.49320, 0.44549]
        ]

    )
)
gradsTotal = (
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
        #print(self.network.thetas)

    def test_correctLayers(self):
        for i, thetas in enumerate(self.network.thetas):
            assert_array_almost_equal(thetas, np.array(thetas[i]), decimal=5)

    def test_labelToOutputs(self):
        label0 = [
            [1.0],
            [0]
        ]
        label1 = [
            [0],
            [1.0]
        ]
        assert_array_almost_equal(self.network.labelToOutputs(0), np.array(label0), decimal=5)
        assert_array_almost_equal(self.network.labelToOutputs(1), np.array(label1), decimal=5)

    def test_outputsToLabel(self):
        label0 = [
            [1.0],
            [0]
        ]
        label1 = [
            [0],
            [1.0]
        ]
        self.assertEqual(self.network.outputsToLabel(label0), 0)
        self.assertEqual(self.network.outputsToLabel(label1), 1)
    
    def test_toVector(self):
        attr1 = [1.0, 0.56, 0]
        assert_array_almost_equal(toVector(attr1), np.array([[1.0], [0.56], [0]]), decimal=5)

    def test_batchGroups(self):
        dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(self.network._batchGroups(dataset), [[1,2,3], [4,5,6], [7,8,9], [10]])

    def test_forwardPropagation(self):
        for example, predicted in zip(examples, predicteds):
            ex = toVector(example)
            pred = toVector(predicted)
            assert_array_almost_equal(self.network.forwardPropagation(ex), pred, decimal=5)

    def test_backPropagation(self):
        for example, output in zip(examples, outputs):
            ex = toVector(example)
            out = toVector(output)
            pred = self.network.forwardPropagation(ex)
            print(self.network.a)
            self.network.backPropagation(pred - out)
        
        for i, D in enumerate(self.network.D):
            assert_array_almost_equal(D, np.array(grads[i]), decimal=5)

    def test_J(self):
        for i, (output, predicted) in enumerate(zip(outputs, predicteds)):
            out = self.network._atributesToInputs(output)
            pred = self.network._atributesToInputs(predicted)
            Jvec = self.network._J(out, pred)
            self.assertAlmostEqual(Jvec.sum(), J[i], places=3)

    def test_cost_computation(self):
        for output, predicted in zip(outputs, predicteds):
            out = self.network._atributesToInputs(output)
            pred = self.network._atributesToInputs(predicted)
            self.network.J += self.network._J(out, pred).sum()
        self.assertAlmostEqual(self.network.computeCost(len(outputs)), Jtotal, places=4)

    def test_trainTurn(self):
        datapoints = [
            (self.network._atributesToInputs(input), self.network._atributesToInputs(output)) 
            for input, output in zip(examples, outputs)
        ]
        err = self.network.trainTurn(datapoints)
        for i, layer in enumerate(self.network.layers):
            grad = layer.D
            assert_array_almost_equal(grad, np.array(gradsTotal[i]), decimal=5)
        # self.assertAlmostEqual(err, np.sum(grad**2), places=5)
        err = self.network.trainTurn(datapoints, updateCost=True)
        self.assertAlmostEqual(self.network.computeCost(len(examples)), Jtotal, places=2)

if __name__ == '__main__':
  main()