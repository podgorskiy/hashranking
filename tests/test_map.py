import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hashranking
import numpy as np
import unittest
from tests import timer


class HammingDistTests(unittest.TestCase):
    def test_numpy_implementation(self):
        rank = np.asarray([[0, 1, 2], [2, 1, 0], [1, 2, 0]])
        s = np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        mAP, p, r = hashranking.numpy_implementation.compute_map_from_rank(rank, s, 0)
        self.assertEqual(mAP, 1)

        rank = np.asarray([[0, 1], [1, 0]])
        s = np.asarray([[1, 0], [1, 0]])
        mAP, p, r = hashranking.numpy_implementation.compute_map_from_rank(rank, s, 0)
        self.assertEqual(mAP, 0.75)

    def test_map_from_rank(self):
        db_size = 1000
        query_size = 200
        class_count = 10
        hash_size = 24
        top_n = 50

        db = np.random.randint(class_count, size=db_size, dtype=np.uint32)
        query = np.random.randint(class_count, size=query_size, dtype=np.uint32)

        hashes_class = np.random.rand(class_count, hash_size).astype(np.float32) - 0.5

        hashes_db = hashes_class[db] + 0.3 * np.random.randn(db_size, hash_size).astype(np.float32)
        hashes_query = hashes_class[query] + 0.3 * np.random.randn(query_size, hash_size).astype(np.float32)

        s = hashranking.numpy_implementation._compute_similarity(db, query)

        rank = hashranking.numpy_implementation.hamming_rank(hashes_query, hashes_db)
        mAP_py, p, r = hashranking.numpy_implementation.compute_map_from_rank(rank, s, top_n)
        
        rank = hashranking.hamming_rank(hashes_query, hashes_db)
        mAP_cpp, p, r = hashranking.compute_map_from_rank(rank, s, top_n)

        self.assertEqual(mAP_py, mAP_cpp)

    def test_on_random_less_than_32(self):
        db_size = 1000
        query_size = 200
        class_count = 10
        hash_size = 24
        top_n = 800

        db = np.random.randint(class_count, size=db_size, dtype=np.uint32)
        query = np.random.randint(class_count, size=query_size, dtype=np.uint32)

        hashes_class = np.random.rand(class_count, hash_size).astype(np.float32) - 0.5

        hashes_db = hashes_class[db] + 0.3 * np.random.randn(db_size, hash_size).astype(np.float32)
        hashes_query = hashes_class[query] + 0.3 * np.random.randn(query_size, hash_size).astype(np.float32)

        s = hashranking.numpy_implementation._compute_similarity(db, query)

        rank = hashranking.numpy_implementation.hamming_rank(hashes_query, hashes_db)
        mAP_py, p, r = hashranking.numpy_implementation.compute_map_from_rank(rank, s, top_n)

        rank = hashranking.hamming_rank(hashes_query, hashes_db)
        mAP_cpp, p, r = hashranking.compute_map_from_rank(rank, s, top_n)

        self.assertEqual(mAP_py, mAP_cpp)

        mAP_py, p, r = hashranking.numpy_implementation.compute_map_from_hashes(hashes_db, hashes_query, db, query, top_n)
        mAP_cpp, p, r = hashranking.compute_map_from_hashes(hashes_db, hashes_query, db, query, top_n)

        self.assertEqual(mAP_py, mAP_cpp)

    def test_on_random_64(self):
        db_size = 1000
        query_size = 200
        class_count = 10
        hash_size = 64
        top_n = 500

        db = np.random.randint(class_count, size=db_size, dtype=np.uint32)
        query = np.random.randint(class_count, size=query_size, dtype=np.uint32)

        hashes_class = np.random.rand(class_count, hash_size).astype(np.float32) - 0.5

        hashes_db = hashes_class[db] + 0.3 * np.random.randn(db_size, hash_size).astype(np.float32)
        hashes_query = hashes_class[query] + 0.3 * np.random.randn(query_size, hash_size).astype(np.float32)

        s = hashranking.numpy_implementation._compute_similarity(db, query)

        rank = hashranking.numpy_implementation.hamming_rank(hashes_query, hashes_db)
        mAP_py, p, r = hashranking.numpy_implementation.compute_map_from_rank(rank, s, top_n)

        rank = hashranking.hamming_rank(hashes_query, hashes_db)
        mAP_cpp, p, r = hashranking.compute_map_from_rank(rank, s, top_n)

        self.assertEqual(mAP_py, mAP_cpp)

        mAP_py, p, r = hashranking.numpy_implementation.compute_map_from_hashes(hashes_db, hashes_query, db, query, top_n)
        mAP_cpp, p, r = hashranking.compute_map_from_hashes(hashes_db, hashes_query, db, query, top_n)

        self.assertEqual(mAP_py, mAP_cpp)

    def test_performance(self):
        db_size = 10000
        query_size = 2000
        class_count = 10
        hash_size = 64
        top_n = 500

        db = np.random.randint(class_count, size=db_size, dtype=np.uint32)
        query = np.random.randint(class_count, size=query_size, dtype=np.uint32)

        hashes_class = np.random.rand(class_count, hash_size).astype(np.float32) - 0.5

        hashes_db = hashes_class[db] + 0.3 * np.random.randn(db_size, hash_size).astype(np.float32)
        hashes_query = hashes_class[query] + 0.3 * np.random.randn(query_size, hash_size).astype(np.float32)

        @timer.timer
        def numpy():
            return hashranking.numpy_implementation.compute_map_from_hashes(hashes_db, hashes_query, db, query, top_n)

        @timer.timer
        def cpp_extension():
            return hashranking.compute_map_from_hashes(hashes_db, hashes_query, db, query, top_n)

        mAP_py, p, r = numpy()
        mAP_cpp, p, r = cpp_extension()

        print("\n")

        print("numpy implementation: %f sec" %(numpy.time))
        print("cpp implementation: %f sec" %(cpp_extension.time))

        print("\n Speedup: x%.2f" % (numpy.time / cpp_extension.time))

        self.assertEqual(mAP_py, mAP_cpp)
