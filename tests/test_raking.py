import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hashranking
import numpy as np
import unittest


class HammingDistTests(unittest.TestCase):
    def test_numpy_implementation(self):
        self.assertTrue((hashranking.numpy_implementation.hamming_rank(
            [[1.0]],
            [[1.0]]) == [[0]]).all())
        self.assertTrue((hashranking.numpy_implementation.hamming_rank(
            [[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]],
            [[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]]) == [[0, 1, 2], [2, 1, 0], [1, 0, 2]]).all())

    def test_cpp_extension(self):
        self.assertTrue((hashranking.hamming_rank(
            [[1.0]],
            [[1.0]]) == [[0]]).all())
        self.assertTrue((hashranking.hamming_rank(
            [[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]],
            [[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]]) == [[0, 1, 2], [2, 1, 0], [1, 0, 2]]).all())

    def test_on_random_less_than_32(self):
        b1 = np.random.rand(200, 24).astype(np.float32) - 0.5
        b2 = np.random.rand(500, 24).astype(np.float32) - 0.5
        d1 = hashranking.hamming_rank(b1, b2)
        d2 = hashranking.numpy_implementation.hamming_rank(b1, b2)

        self.assertTrue((d1 == d2).all())

    def test_on_random_32(self):
        b1 = np.random.rand(200, 32).astype(np.float32) - 0.5
        b2 = np.random.rand(500, 32).astype(np.float32) - 0.5
        d1 = hashranking.hamming_rank(b1, b2)
        d2 = hashranking.numpy_implementation.hamming_rank(b1, b2)

        self.assertTrue((d1 == d2).all())

    def test_on_random_greater_than_32(self):
        b1 = np.random.rand(200, 48).astype(np.float32) - 0.5
        b2 = np.random.rand(500, 48).astype(np.float32) - 0.5
        d1 = hashranking.hamming_rank(b1, b2)
        d2 = hashranking.numpy_implementation.hamming_rank(b1, b2)

        self.assertTrue((d1 == d2).all())

    def test_on_random_64(self):
        b1 = np.random.rand(200, 64).astype(np.float32) - 0.5
        b2 = np.random.rand(500, 64).astype(np.float32) - 0.5
        d1 = hashranking.hamming_rank(b1, b2)
        d2 = hashranking.numpy_implementation.hamming_rank(b1, b2)

        self.assertTrue((d1 == d2).all())
