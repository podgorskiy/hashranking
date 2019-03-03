import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hashranking
import numpy as np
import unittest


class ArgSortTests(unittest.TestCase):
    def test_basic(self):
        self.assertTrue((hashranking.argsort([[1, 2], [1, 3]]) == [[0, 1], [0, 1]]).all())
        self.assertTrue((hashranking.argsort([[2, 1], [3, 1]]) == [[1, 0], [1, 0]]).all())
        self.assertTrue((hashranking.argsort([[2, 1], [1, 3]]) == [[1, 0], [0, 1]]).all())

    def test_on_random(self):
        dist = np.random.randint(0, 64, size=(500, 1000), dtype=np.uint8)
        self.assertTrue((hashranking.argsort(dist) == np.argsort(dist, 1, kind='mergesort')).all())

