from unittest import TestCase, main
import cross_validation
import data
import math
import statistics

NaN = float('nan')

class OneClassifier(object):
    def train(self, dataset):
        pass

    def classify(self, datapoint):
        return 1

class ZeroClassifier(object):
    def train(self, dataset):
        pass

    def classify(self, datapoint):
        return 0

class AlwaysWrongClassifier(object):
    def train(self, dataset):
        pass

    def classify(self, datapoint):
        return not datapoint.label

class AlwaysRightClassifier(object):
    def train(self, database):
        pass

    def classify(self, datapoint):
        return datapoint.label

class CrossValidatorTest(TestCase):
    def setUp(self):
        self.zero_dataset = [data.Datapoint([1], 0)]*8
        self.mix_dataset = [data.Datapoint([1], 0)]*4 + [data.Datapoint([1], 1)]*4

    def test_evaluation_parameters(self):
        test_cases = [
            (AlwaysRightClassifier, self.mix_dataset, {'errors': 0, 'accuracies': 1, 'recalls': 1, 'precisions': 1, 'f1s': 1}),
            (AlwaysWrongClassifier, self.mix_dataset, {'errors': 1, 'accuracies': 0, 'recalls': 0, 'precisions': 0, 'f1s': NaN}),
            (ZeroClassifier, self.zero_dataset, {'errors': 0, 'accuracies': 1, 'recalls': 1, 'precisions': 1, 'f1s': 1}),
            (OneClassifier, self.zero_dataset, {'errors': 1, 'accuracies': 0, 'recalls': 0, 'precisions': NaN, 'f1s': NaN}),
            (ZeroClassifier, self.mix_dataset, {'errors': 0.5, 'accuracies': 0.5, 'recalls': 1.0, 'precisions': 0.5, 'f1s': (1.0/1.5)}),
        ]

        for Classifier, dataset, parameters in test_cases:
            cv = cross_validation.CrossValidator(2, Classifier())
            cv.run(dataset)
            for n, v in parameters.items():
                got = statistics.mean(getattr(cv, n)(0))
                if math.isnan(v):
                    self.assertTrue(math.isnan(got))
                else:
                    self.assertAlmostEqual(v, got, places=6)

    def test_decision_matrix_building(self):
        cv = cross_validation.CrossValidator(None, None)
        test_cases = [
            # predicted, actual, expected_matrix
            (
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]
                ]
            ),

            (
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 0],
                [
                    [3, 0],
                    [2, 1],
                ]
            ),

        ]

        for predicted, actual, expected in test_cases:
            got = cv._build_confusion_matrix(predicted, actual)
            self.assertEqual(expected, got)

    def test_confusion_matrix_metrics(self):
        cv = cross_validation.CrossValidator(None, None)
        test_cases = [
            # matrix, tp, fp, tn, fn, n
            ([
                [1, 1],
                [1, 1],
            ],
            1, 1, 1, 1, 4),

            ([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            1, 5, 28, 11, 45),

            ([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            1, 2, 4, 2, 9),
        ]

        for matrix, tp, fp, tn, fn, n in test_cases:
            self.assertEqual(tp, cv._tp(matrix, 0))
            self.assertEqual(fp, cv._fp(matrix, 0))
            self.assertEqual(tn, cv._tn(matrix, 0))
            self.assertEqual(fn, cv._fn(matrix, 0))
            self.assertEqual(n, cv._n(matrix))


main()
