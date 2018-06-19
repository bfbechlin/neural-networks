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
