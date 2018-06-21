import folding
import data
import statistics

def nanable(method):
    '''
    Decorates a method so that, if it throws a ZeroDivisionError, it actually
    returns NaN.
    '''
    def exception_catcher(*args):
        try:
            return method(*args)
        except ZeroDivisionError:
            return float('nan')

    return exception_catcher

class CrossValidator(object):
    def __init__(self, k, classifier):
        self.k = k
        self.classifier = classifier

    def run(self, dataset):
        '''
        Run k cross-validation rounds so that metrics can be calculated.
        '''
        self._matrices = []
        i = 0
        for t, v in self._test_and_validation_sets(dataset):
            i += 1
            self.classifier.train(t)
            predicted = [self.classifier.classify(p) for p in v]
            matrix = self._build_confusion_matrix(predicted, list(data.labels(v)))
            self._matrices.append(matrix)

    def _test_and_validation_sets(self, dataset):
        '''
        Return the CV test and validation sets. That is, for i = 1..k, return
        the pair (folds[i], rest of folds) as (test, train) sets.
        '''
        folds = folding.fold(self.k, dataset)
        for i in range(self.k):
            yield folds[i], self._flatten(folds[:i] + folds[i+1:])

    def _flatten(self, a_list):
        '''
        Turns [[1, 2, 3], [4, 5], [6, 7]] into [1, 2, 3, 4, 5, 6, 7].
        '''
        return [i for s in a_list for i in s]

    def _build_confusion_matrix(self, predicted, actual):
        all_labels = set(predicted).union(set(actual))
        matrix = self._create_matrix(len(all_labels))
        for p, a in zip(predicted, actual):
            matrix[p][a] += 1

        return matrix

    def f1s(self, label):
        '''
        Get a list consisting of the F1 measure for each of the CV-rounds,
        considering label as a positive and anything else as a negative.
        '''
        f1 = []
        for p, r in zip(self.precisions(label), self.recalls(label)):
            f1.append(self._compute_f1(p, r))
        return f1

    @nanable
    def _compute_f1(self, precision, recall):
        return 2*precision*recall/(precision + recall)

    def errors(self, label):
        '''
        Get a list consisting of the error for each of the CV-rounds,
        considering label as a positive and anything else as a negative.
        '''
        e = []
        for a in self.accuracies(label):
            e.append(1 - a)
        return e

    def accuracies(self, label):
        '''
        Get a list consisting of the accuracy for each of the CV-rounds,
        considering label as a positive and anything else as a negative.
        '''
        a = []
        for m in self._matrices:
            a.append(self._accuracy(m, label))
        return a

    @nanable
    def _accuracy(self, matrix, label):
        return (self._tp(matrix, label) + self._tn(matrix, label))/self._n(matrix)

    def precisions(self, label):
        '''
        Get a list consisting of the precision for each of the CV-rounds,
        considering label as a positive and anything else as a negative.
        '''
        p = []
        for m in self._matrices:
            p.append(self._precision(m, label))
        return p

    @nanable
    def _precision(self, matrix, label):
        return self._tp(matrix, label)/(self._tp(matrix, label) + self._fp(matrix, label))

    def recalls(self, label):
        '''
        Get a list consisting of the recall for each of the CV-rounds,
        considering label as a positive and anything else as a negative.
        '''
        r = []
        for m in self._matrices:
            r.append(self._recall(m, label))
        return r

    @nanable
    def _recall(self, matrix, label):
        return self._tp(matrix, label)/(self._tp(matrix, label) + self._fn(matrix, label))

    def _fp(self, matrix, label):
        fp = 0
        for l, c in enumerate(matrix[label]):
            if l == label: # same column is the true positive
                continue
            fp += c
        return float(fp)

    def _fn(self, matrix, label):
        fn = 0
        for l, r in enumerate(matrix):
            if l == label: # same row is the true positive
                continue
            fn += r[label]
        return float(fn)

    def _tp(self, matrix, label):
        return float(matrix[label][label])

    def _tn(self, matrix, label):
        return float(self._n(matrix) - self._tp(matrix, label) - self._fp(matrix, label) - self._fn(matrix, label))

    def _n(self, matrix):
        '''
        Return the number of datapoints given a confusion matrix.
        '''
        return float(sum(c for c in [sum(r) for r in matrix]))

    def _create_matrix(self, n):
        '''
        Return an n by n matrix filled with zeros.
        '''
        return [[0]*n for i in range(n)]

