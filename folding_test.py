from unittest import TestCase, main
import data
import folding

dp0 = data.Datapoint([1.0], 0)
dp1 = data.Datapoint([1.0], 1)
dp2 = data.Datapoint([1.0], 2)

class TestFolding(TestCase):
    def test_folds_are_of_the_expected_size(self):
        test_cases = [
            # dataset, k, fold_lens
            ([dp0, dp0], 2, (1, 1)),
            ([dp0, dp0, dp0], 2, (2, 1)),
            ([dp0, dp0, dp0, dp0], 2, (2, 2)),
            ([dp0, dp0, dp0, dp1], 2, (3, 1)),
            ([dp0, dp0, dp0, dp1, dp1], 2, (3, 2)),
            ([dp0, dp0, dp1, dp1, dp1], 2, (3, 2)),
        ]

        for dataset, k, fold_lens in test_cases:
            folds = folding.fold(k, dataset)
            self.assertEqual(k, len(folds))
            self.assertEqual(fold_lens, tuple(map(len, folds)))

    def test_folds_are_stratified(self):
        test_cases = [
            # dataset, k
            ([dp1]*2 + [dp0]*2, 2),
            ([dp1]*3 + [dp0]*6, 3),
            ([dp1]*10 + [dp0]*90, 10),
            ([dp2]*2 + [dp1]*2 + [dp0]*2, 2),
        ]

        for dataset, k in test_cases:
            folds = folding.fold(k, dataset)
            label_counts = [[self.count_label(f, l) for l in set(data.labels(dataset))] for f in folds]
            self.assertEqual(label_counts[1:], label_counts[:-1])

    def count_label(self, dataset, label):
        filtered = filter(lambda l: l == label, data.labels(dataset))
        return len(list(filtered))

main()
