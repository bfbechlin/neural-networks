import collections

reader_parameters = {}

def benchmark_label_to_int(label):
    return {'nao': 0, 'sim': 1}[label]

reader_parameters['benchmark'] = {
    'label_column': 4,
    'cast_functions': [str]*4 + [benchmark_label_to_int]
}

reader_parameters['pima'] = {
    'label_column': 8,
    'cast_functions': [float]*8 + [int]
}

def wine_label_to_int(label):
    return int(label) - 1

reader_parameters['wine'] = {
    'label_column': 0,
    'cast_functions': [wine_label_to_int] + [float]*13,
}

def ionosphere_label_to_int(label):
    return {'b': 0, 'g': 1}[label]

reader_parameters['ionosphere'] = {
    'label_column': 34,
    'cast_functions': [float]*34 + [ionosphere_label_to_int]
}

def cancer_label_to_int(label):
    return {'B': 0, 'M': 1}[label]

reader_parameters['cancer'] = {
    'label_column': 1,
    'id_column': 0,
    'cast_functions': [None, cancer_label_to_int] + [float]*30
}

def majority_vote(l):
    '''
    Return the most common element in l.
    '''
    c = collections.Counter(l)
    return c.most_common()[0][0]

def decode_matrix_list(file_name):
    result = []
    with open(file_name) as f:
        for line in f:
            rows = line.split(';')
            rowList = []
            for row in rows:
                numbers = row.split(',')
                rowList.append([float(n) for n in numbers])
            result.append(rowList)
    return result

def encode_matrix_list(matrix_list):
    string = ''
    for k, matrix in enumerate(matrix_list):
        for j, line in enumerate(matrix):
            for i, number in enumerate(line):
                string += '{0:.5f}'.format(round(number,5))
                if i != len(line) - 1:
                    string += ', '
            if j != len(matrix) - 1:
                string += '; '
        if k != len(matrix_list) - 1:
            string += '\n'
    return string 

def decode_network(file_name):
    LAMBDA = 0
    network = []
    with open(file_name) as f:
        for i, line in enumerate(f):
            if i == 0:
                LAMBDA = float(line)
            else:
                network.append(int(line))
    return network, LAMBDA