from neural_layer import NeuralLayer

class NeuralNetwork:
    def __init__(self, layers):
        self.layer = (
            NeuralLayer(inputs=layers[i], outputs=layers[i+1]) for i in range(0, len(layers)-1)
        )

    def train(self):
        pass
    
    def classify(self):
        pass