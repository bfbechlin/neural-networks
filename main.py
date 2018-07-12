from neural_network import NeuralNetwork
import data
import random
import utils
import sys
from cross_validation import CrossValidator
from folding import fold
from statistics import mean, stdev

random.seed(0)

parameters = {}
parameters['pima'] = {
    'STOP': 1000,
    'ALPHA': 0.5,
    'K': 768 / 10,
    'NETWORKS': [
        [8, 4, 2],
        [8, 4, 4, 2],
    ]
}
parameters['cancer'] = {
    'STOP': 250,
    'ALPHA': 0.1,
    'K': 569 / 10,
    'NETWORKS': [
        [30, 8, 2],
        [30, 4, 4, 2],
    ]
}
parameters['wine'] = {
    'STOP': 500,
    'ALPHA': 0.1,
    'K': 178 / 10,
    'NETWORKS': [
        [13, 8, 3],
        [13, 4, 4, 3],
    ]
}
parameters['ionosphere'] = {
    'STOP': 250,
    'ALPHA': 0.1,
    'K': 351 / 10,
    'NETWORKS': [
        [34, 8, 2],
        [34, 4, 4, 2],
    ]
}

def run(dataset_name, output_file_name):
    dataset = data.read_dataset('datasets/' + dataset_name + '.csv', **utils.reader_parameters[dataset_name])
    random.shuffle(dataset)
    params = parameters[dataset_name]
    network = params['NETWORKS'][0]

    with open('J_vs_N_' + str(network) + dataset_name + '.csv', 'w') as file:
        folds = fold(10, dataset)
        valids = folds[0]
        train = [i for s in folds[1:] for i in s]
        trains = [train[i:i + 10] for i in range(0, len(train), 10)]
        n = 0
        net = NeuralNetwork(network, ALPHA=params['ALPHA'], LAMBDA=0, K=0, BETA=0.8, STOP=1)
        for i in range(50):
            for train in trains:
                n += len(train)
                net.train(train)
                J = 0
                for valid in valids:
                    J += net.verifyPerformance(valid)
                file.write(str(n) + ',' + str(J) +'\n')

    for network in params['NETWORKS']:
        with open('f1_vs_lambda_' + str(network) + dataset_name + '.csv', 'w') as file:
            for LAMBDA in [0, 1, 8, 64, 512]:
                net = NeuralNetwork(network, ALPHA=params['ALPHA'], LAMBDA=LAMBDA/1000.0, K=params['K'], BETA=0.8, STOP=params['STOP'])
                cv = CrossValidator(k=10, classifier=net)
                cv.run(dataset)
                try:
                    file.write(str(LAMBDA/1000.0) + ',' + str(mean(cv.f1s(1))) + ',' + str(stdev(cv.f1s(1))) +'\n')
                except:
                    file.write(str(LAMBDA/1000.0) + ',0,0\n')
                file.flush()
   

for dataset_name in ['wine', 'ionosphere', 'cancer', 'pima']:
    run(dataset_name, 'f1_vs_lambda_'+dataset_name+'.csv')

