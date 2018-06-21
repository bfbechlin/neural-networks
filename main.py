from neural_network import NeuralNetwork
import data
import random
import utils
import sys
from cross_validation import CrossValidator
from statistics import mean, stdev

random.seed(0)

def run(dataset_name, output_file_name):
    dataset = data.read_dataset('datasets/' + dataset_name + '.csv', **utils.reader_parameters[dataset_name])
    random.shuffle(dataset)
    inputs = len(dataset[0].attributes)
    outputs = max(data.labels(dataset)) + 1

    with open(output_file_name, 'w') as file:
        for lam in range(0, 10):
            network = NeuralNetwork([inputs, 8, outputs], ALPHA=0.1, LAMBDA=lam/10.0, K=100, BETA=0.8, STOP=500)
            cv = CrossValidator(k=10, classifier=network)
            cv.run(dataset)
            try:
                file.write(str(lam/100.0) + ',' + str(mean(cv.f1s(1))) + ',' + str(stdev(cv.f1s(1))) +'\n')
            except:
                file.write(str(lam/100.0) + ',0,0\n')
            file.flush()

for dataset_name in ['pima']:
    run(dataset_name, 'f1_vs_lambda_'+dataset_name+'.csv')

