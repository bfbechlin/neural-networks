from collections import defaultdict
import data

def fold(k, dataset):
    '''
    Return the stratified folds of the dataset.

    The difference between the number of instances
    in one fold and another is at most the number of labels in the dataset. The
    difference between the number of intances of a particular label in one fold
    and another is at most 1.

    The algorithm works like this:
    For each label, give one datapoint to each fold until you run out of
    datapoints. To do this in one pass, keep a dictionary that maps a label to
    the next fold to receive that label.
    '''
    folds = [[] for i in range(k)]
    # next_receiver[label] is the next fold to receive that label
    next_receiver = defaultdict(int)
    for p, l in data.datapoints_and_labels(dataset):
        i = next_receiver[l]
        folds[i].append(p)
        next_receiver[l] = (i + 1)%k

    return folds
