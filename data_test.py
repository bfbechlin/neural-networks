from unittest import TestCase, main
import data

class TestData(TestCase):
    def setUp(self):
        self.attributes = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]

        self.labels = [
            1,
            2,
            13,
            0,
        ]

        self.dataset = [data.Datapoint(a, l) for a, l in zip(self.attributes, self.labels)]


    def test_attributes_iterator(self):
        got = list(data.attributes(self.dataset))
        self.assertEqual(self.attributes, got)

    def test_labels_iterator(self):
        got = list(data.labels(self.dataset))
        self.assertEqual(self.labels, got)

    def test_attributes_and_labels_iterator(self):
        got = list(data.attributes_and_labels(self.dataset))
        expected = list(zip(self.attributes, self.labels))
        self.assertEqual(expected, got)

main()
