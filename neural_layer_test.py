from unittest import TestCase, main
from neural_layer import NeuralLayer
from numpy import array, vstack
from numpy.testing import assert_array_almost_equal

class NeuralLayerTest(TestCase):
  def setUp(self):
    self.ex1 = (
      NeuralLayer(1, 2, [[0.4, 0.1], [0.3, 0.2]]),
      NeuralLayer(2, 1, [[0.7, 0.5, 0.6]]),
    )

  def addBais(self, inputs):
    return vstack((array(1.), array(inputs)))

  def test_activation_forward(self):
    results = (
      [[0.13]], 
      [[0.601807], [0.5807858]], 
      [[0.794027]]
    )
    for i, layer in enumerate(self.ex1):
      activations = layer.computeActivations(array(results[i]))
      spectation = array(results[i+1])
      assert_array_almost_equal(activations, spectation)
      assert_array_almost_equal(layer.a, spectation)

  def test_delta_computation(self):
    examples1 = [
      {
        'x': 0.13000, 
        'p': 0.794027, 
        'y': 0.90000,
        'D': (
          [ 
            [-0.01270,  -0.00165],
            [-0.01548,  -0.00201]
          ],
          [
            [-0.10597,  -0.06378,  -0.06155]
          ]
        ),
        'deltas': (
          [],
          [
            [-0.01270], 
            [-0.01548]
          ],
          [
            [-0.10597]
          ]
        )
      }, 
      {
        'x': 0.42000, 
        'p': 0.795966, 
        'y': 0.23000,
        'D': (
          [
            [0.06740, 0.02831],
			      [0.08184, 0.03437]
          ],
          [
            [0.56597,  0.34452,  0.33666]
          ]
        ),
        'deltas': (
          [],
          [
            [0.06740], 
            [0.08184]
          ],
          [
            [0.56597]
          ]
        )
      }
    ]

    grads1 = [
      [
        [0.02735, 0.01333],
			  [0.03318, 0.01618]
      ],
      [
        [0.23000, 0.14037, 0.13756]
      ]
    ]

    for example in examples1:
      activations = array([[example['x']]])
      for i, layer in enumerate(self.ex1):
        activations = layer.computeActivations(activations)
        layer.DSpec = array(example['D'][i])
        layer.deltaSpec = array(example['deltas'][i])
        layer.gradSpec = array(grads1[i])
      assert_array_almost_equal(activations, array([[example['p']]]), decimal=5)
      delta = activations - array(example['y'])
      assert_array_almost_equal(delta, array(example['deltas'][-1]), decimal=5)

      for i, layer in enumerate(reversed(self.ex1)):
        D = layer.computeAndUpdateGrads(delta)
        assert_array_almost_equal(D, layer.DSpec, decimal=5)
        if i != len(self.ex1) - 1:
          delta = layer.computeDeltas(delta)
          assert_array_almost_equal(delta, layer.deltaSpec, decimal=5)

      for i, layer in enumerate(self.ex1):
        assert_array_almost_equal(layer.D, layer.gradSpec, decimal=5)

if __name__ == '__main__':
  main()