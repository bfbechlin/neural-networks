from collections import namedtuple

Datapoint = namedtuple('Datapoint', ['attributes', 'label'])

def attributes(dataset):
    '''
    Return an iterator of the attributes in the dataset.
    '''
    for p in dataset:
        yield p.attributes

def labels(dataset):
    '''
    Return an iterator of the labels in the dataset.
    '''
    for p in dataset:
        yield p.label

def attributes_and_labels(dataset):
    '''
    Return an iterator of the attributes and labels in the dataset.
    '''
    for a, l in zip(attributes(dataset), labels(dataset)):
        yield a, l

def single_attribute(dataset, index):
    '''
    Return an iterator of the attribute fixed.
    '''
    if(index >= len(dataset[0].attributes) or index < 0):
        print(index)
        raise Exception('Attribute selected does not exists')
    for datapoint in dataset:
        yield datapoint.attributes[index]

def datapoints_and_labels(dataset):
    '''
    Return an iterator of the datapoints and labels in the dataset.
    '''
    for p, l in zip(dataset, labels(dataset)):
        yield p, l

def read_dataset(filename, id_column=None, label_column=-1, cast_functions=None):
    '''
    Return a dataset (list of Datapoints) based on the given CSV file. The first
    row must be of headers and will be used to figure the number of columns.
    label is the column number for the labels and ident is the column number of
    the identifier, which will be ignored. cast_functions is a list of functions
    used to cast each row. For example, if

    cast_functions == [None, int, float, str, lambda x: 0 if x == 'a' else 1]

    then id_column must be 0, the second column will be cast to integers, the
    third to floats and the fourth to strings. The last will be 0 if the value
    is 'a' and 1 otherwise. This can be used to cast labels which are not
    numeric. All float attributes will be normalized.
    '''
    with open(filename) as csv:
        header = csv.readline().strip().split(',')
        num_columns = len(header)

        maxs = [float('-inf')]*num_columns
        mins = [float('+inf')]*num_columns
        label_column = label_column%num_columns

        rows = []
        for row in csv:
            row = row.strip().split(',')

            for i, c in enumerate(row):
                if i != id_column:
                    row[i] = cast_functions[i](c)

                if type(row[i]) == float:
                    maxs[i] = max(maxs[i], row[i])
                    mins[i] = min(mins[i], row[i])

            rows.append(row)

    dataset = []
    for row in rows:
        label = row[label_column]
        attributes = []
        for i, c in enumerate(row):
            if i == label_column or i == id_column:
                continue

            if type(row[i]) == float and maxs[i] != mins[i]:
                row[i] = (c - mins[i])/(maxs[i] - mins[i])

            attributes.append(row[i])

        dataset.append(Datapoint(attributes, label))

    return dataset
