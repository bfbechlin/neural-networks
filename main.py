from neural_network import NeuralNetwork
import data
import random
import utils
import sys

random.seed(0)

def run(dataset_name, output_file, num_trees_range):
    '''
    Run 10 fold cross validation a dataset varying the num_trees parameter
    according to the iterator num_trees_range. Output to the given file.
    '''
    dataset = data.read_dataset('datasets/' + dataset_name + '.csv', **utils.reader_parameters[dataset_name])
    random.shuffle(dataset)
    inputs = len(dataset[0].attributes)
    outputs = max(data.labels(dataset)) + 1
    network = NeuralNetwork([inputs, 10, 5, 10, outputs], ALPHA=5, STOP=1.3, LAMBDA=0.25)
    network.train(dataset)
    errors = 0
    for datapoint in dataset:
        if network.classify(datapoint) != datapoint.label:
            errors += 1
    print(errors, errors * 1.0 /len(dataset))

for dataset_name in ['pima']:

    run(dataset_name, 'f1-vs-num-trees-' + dataset_name + '.csv', [1, 2, 4, 8, 16, 32, 64, 128])
