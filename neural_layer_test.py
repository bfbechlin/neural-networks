from unittest import TestCase, main
from neural_layer import NeuralLayer
from numpy import array
from numpy.testing import assert_array_almost_equal

class NeuralLayerTest(TestCase):
  def setUp(self):
    self.ex1 = (
      NeuralLayer(1, 2, [[0.4, 0.1], [0.3, 0.2]]),
      NeuralLayer(2, 1, [0.7, 0.5, 0.6]),
    )

  def test_activation_forward(self):
    results = [[0.13], [[0.601807], [0.5807858]], [0.794027]]
    for i, layer in enumerate(self.ex1):
      activations = layer.computeActivations(array(results[i]))
      spectation = array(results[i+1])
      assert_array_almost_equal(activations, spectation)
      assert_array_almost_equal(layer.a, spectation)

  def test_delta_computation(self):
    examples = [
      {'x': 0.13000, 'p': 0.794027, 'y': 0.90000}, 
      {'x': 0.42000, 'p': 0.795966, 'y': 0.23000}
    ]

    for example in examples:
      activations = array(example['x'])
      for layer in self.ex1:
        activations = layer.computeActivations(activations)
      assert_array_almost_equal(activations, array([example['p']]))
      delta = array(example['y'])
      for i, layer in enumerate(reversed(self.ex1)):
        print(delta)
        delta = layer.computeDelta(delta, i == 0)
        print(delta)


if __name__ == '__main__':
  main()