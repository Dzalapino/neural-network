from unittest import TestCase
import numpy as np


class TestDataset(TestCase):
    def setUp(self):
        from neural_network.dataset import Dataset
        self.dataset = Dataset(path='../resources/Iris.csv', training_ratio=0.8, has_index_col=True, if_shuffle=False)
        self.df = self.dataset.df
        self.train_X = self.dataset.train_X
        self.train_y = self.dataset.train_y
        self.eval_X = self.dataset.eval_X
        self.eval_y = self.dataset.eval_y

    def test_load_dataset(self):
        # Check if the train set size is correct with given training ratio
        self.assertEqual(len(self.train_X), 120)
        # Check if the evaluation set size is correct with given training ratio
        self.assertEqual(len(self.eval_X), 30)
        # Check if the number of feature columns is correct
        self.assertEqual(len(self.train_X[0]), 4)
        # Check if the number of label hot encoded columns is correct
        self.assertEqual(len(self.eval_y[0]), 3)
        # Check if the train features are correct
        self.assertTrue(np.allclose(self.train_X[0], [5.1, 3.5, 1.4, 0.2]))
        # Check if the train hot encoded labels are correct
        self.assertTrue(np.allclose(self.train_y[0], [1, 0, 0]))
        # Check if the evaluation features are correct
        self.assertTrue(np.allclose(self.eval_X[-1], [5.9, 3.0, 5.1, 1.8]))
        # Check if the evaluation labels are correct
        self.assertTrue(np.allclose(self.eval_y[-1], [0, 0, 1]))
