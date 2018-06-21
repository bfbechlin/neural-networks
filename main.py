from new_net import NeuralNetwork
import data
import random
import utils
import sys
from cross_validation import CrossValidator
from statistics import mean, stdev

random.seed(0)

def run(dataset_name, output_file, num_trees_range):
    '''
    Run 10 fold cross validation a dataset varying the num_trees parameter
    according to the iterator num_trees_range. Output to the given file.
    '''
    dataset = data.read_dataset('datasets/' + dataset_name + '.csv', **utils.reader_parameters[dataset_name])
    random.shuffle(dataset)
    random.shuffle(dataset)
    inputs = len(dataset[0].attributes)
    outputs = max(data.labels(dataset)) + 1

    network = NeuralNetwork([inputs, 8, outputs], ALPHA=0.1, STOP=0.1, LAMBDA=0.25, K=100)
    cv = CrossValidator(k=5, classifier=network)
    cv.run(dataset)
    print mean(cv.f1s(1))
    print mean(cv.accuracies(1))
    print mean(cv.precisions(1))
    print mean(cv.recalls(1))
    print mean(cv.errors(1))

for dataset_name in ['wine']:

    run(dataset_name, 'f1-vs-num-trees-' + dataset_name + '.csv', [1, 2, 4, 8, 16, 32, 64, 128])
