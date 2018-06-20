from unittest import TestCase, main
from neural_layer import NeuralLayer
from numpy import array, vstack, subtract
from numpy.testing import assert_array_almost_equal

examples = (
    [
        [0.13000]   
    ],
    [
        [0.42000]
    ]
)
outputs = (
    [
        [0.90000]        
    ],
    [
        [0.23000]
    ]
)
thetas = (
    [
        [0.40000, 0.10000],
	    [0.30000, 0.20000]
    ],
    [
        [0.70000, 0.50000, 0.60000]
    ]
)
activations = (
    (
        [
            [0.13000]
        ],
        [
            [0.60181],
            [0.58079]
        ],
        [
            [0.79403]
        ]
    ),
    (
        [
            [0.42000]
        ],
		
        [
            [0.60874],
            [0.59484]
        ],
        [
            [0.79597]
        ]
    )
)
predicteds = (
    [
        [0.79403]
    ],
    [
        [0.79597]
    ]
)
grads = (
    (
        [
            [-0.01270, -0.00165],
	        [-0.01548, -0.00201]
        ],
        [
            [-0.10597, -0.06378, -0.06155]
        ]
    ),
    (
        [
            [0.06740, 0.02831],
		    [0.08184, 0.03437]
        ],
        [
            [0.56597, 0.34452, 0.33666]
        ]
    )
)
deltas = (
    (
        [
            [-0.01270],
            [-0.01548]
        ],
        [
            [-0.10597]
        ]
    ),
    (
        [
            [0.06740],
            [0.08184]
        ],
        [
            [0.56597]
        ]
    )
)
gradsTotal = (
    [
        [0.02735, 0.01333],
		[0.03318, 0.01618]
    ],
    [
        [0.23000, 0.14037, 0.13756]
    ]  
)

class NeuralLayerTest(TestCase):
    def setUp(self):
        self.layers = (
            NeuralLayer(1, 2, thetas[0]),
            NeuralLayer(2, 1, thetas[1]),
        )

    def addBais(self, inputs):
        return vstack((array(1.), array(inputs)))

    def test_activation_forward(self):
        for i, layer in enumerate(self.layers):
            act = layer.computeActivations(array(activations[0][i]))
            spec = array(activations[0][i+1])
            assert_array_almost_equal(act, spec, decimal=5)

    def test_delta_and_grads(self):
        for i, example in enumerate(examples):
            act = array(example)
            for j, layer in enumerate(self.layers):
                layer.gradSpec = array(grads[i][j])
                layer.deltaSpec = array(deltas[i][j])

                assert_array_almost_equal(act, array(activations[i][j]), decimal=5)
                act = layer.computeActivations(act)
                
            delta = act - array(outputs[i])
            assert_array_almost_equal(delta, array(deltas[i][-1]), decimal=5)

            for j, layer in enumerate(reversed(self.layers)):
                grad = layer.computeAndUpdateGrads(delta)
                assert_array_almost_equal(grad, layer.gradSpec, decimal=5)
                if j != len(self.layers) - 1:
                    assert_array_almost_equal(delta, layer.deltaSpec, decimal=5)
                    delta = layer.computeDeltas(delta)
                    
        for i, layer in enumerate(self.layers):
            assert_array_almost_equal(layer.updateThetas(len(examples), 1, 0), 
                gradsTotal[i], decimal=5)
            assert_array_almost_equal(layer.thetas, subtract(thetas[i], gradsTotal[i]), decimal=5)


if __name__ == '__main__':
    main()
