# Copyright 2017 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Methods to work with hashes"""

import numpy as np
import _hashranking
import time


def timer(f):
    """Decorator for timeing method execution time"""
    def __wrapper(*args, **kw):
        time_start = time.time()
        result = f(*args, **kw)
        time_end = time.time()
        print('func:%r  took: %2.4f sec' % (f.__name__, time_end - time_start))
        return result
    return __wrapper


@timer
def calc_hamming_dist(b1, b2):
    """Compute the hamming distance between every pair of data points represented in each row of b1 and b2"""
    p1 = np.sign(b1).astype(np.int8)
    p2 = np.sign(b2).astype(np.int8)

    r = p1.shape[1]
    d = (r - np.matmul(p1, np.transpose(p2))) // 2
    return d


@timer
def calc_hamming_dist_fast(b1, b2):
    return _hashranking.calc_hamming_dist(b1, b2)


@timer
def calc_hamming_rank(b1, b2):
    """Return rank of pairs. Takes vector of hashes b1 and b2 and returns correspondence rank of b1 to b2
    """
    dist_h = calc_hamming_dist(b1, b2)
    return np.argsort(dist_h, 1, kind='stable')


@timer
def calc_hamming_rank_fast(b1, b2):
    """Return rank of pairs. Takes vector of hashes b1 and b2 and returns correspondence rank of b1 to b2
    """
    dist_h = calc_hamming_dist_fast(b1, b2)
    return _hashranking.argsort(dist_h)


@timer
def compute_map(hashes_db, hashes_query, labels_db, labels_query, top_n=0):
    """Compute MAP for given set of hashes and labels"""
    rank = calc_hamming_rank(hashes_query, hashes_db)
    s = __compute_s(labels_db, labels_query)
    return __calc_map(rank, s, top_n)

@timer
def compute_map_fast(hashes_db, hashes_query, labels_db, labels_query, top_n=0):
    """Compute MAP for given set of hashes and labels"""
    rank = calc_hamming_rank_fast(hashes_query, hashes_db)
    return _hashranking.calc_map(rank, labels_db, labels_query, top_n)

@timer
def compute_map_faster(hashes_db, hashes_query, labels_db, labels_query, top_n=0):
    """Compute MAP for given set of hashes and labels"""
    return _hashranking.calc_map_from_hashes(hashes_db, hashes_query,  labels_db, labels_query, top_n)


def __compute_s(labels_db, labels_query, and_mode=False):
    """Return similarity matrix between two label vectors
    The output is binary matrix of size n_train x n_test
    """
    if and_mode:
        return np.bitwise_and(labels_db, np.transpose(labels_query)).astype(dtype=np.bool)
    else:
        return np.equal(labels_db, labels_query[:, np.newaxis])


@timer
def __calc_map(rank, s, top_n):
    """compute mean average precision (MAP)"""
    Q, N = s.shape
    if top_n == 0:
        top_n = N
    pos = np.asarray(range(1, top_n + 1), dtype=np.float32)
    mAP = 0
    av_precision = np.zeros(top_n)
    av_recall = np.zeros(top_n)
    for q in range(Q):
        total_number_of_relevant_documents = np.sum(s[q])
        relevance = s[q, rank[q, :top_n]]
        cumulative = np.cumsum(relevance)
        number_of_relative_docs = cumulative[-1]
        if number_of_relative_docs != 0:
            precision = cumulative.astype(np.float32) / pos
            recall = cumulative / total_number_of_relevant_documents
            av_precision += precision
            av_recall += recall
            ap = np.dot(precision.astype(np.float64), relevance)
            ap /= number_of_relative_docs
            mAP += ap
    mAP /= Q
    av_precision /= Q
    av_recall /= Q

    return float(mAP), av_precision, av_recall


# For testing
if __name__ == '__main__':
    b1 = np.random.rand(2000, 64).astype(np.float32) - 0.5
    b2 = np.random.rand(5000, 64).astype(np.float32) - 0.5
    # d1_ = calc_hamming_dist_fast(b1, b2)
    # d2_ = calc_hamming_dist(b1, b2)
    # print(d1_)
    # print(d2_)
    # print("Passed!" if (d1_ == d2_).all() else "Failed!")

    d1 = calc_hamming_rank_fast(b1, b2)
    d2 = calc_hamming_rank(b1, b2)
    print(d1)
    print(d2)
    print(d1.shape)
    print(d2.shape)

    print("Passed!" if (d1 == d2).all() else "Failed!")

    db_size = 10000
    query_size = 2000
    class_count = 10
    hash_size = 64

    db = np.random.randint(class_count, size=db_size, dtype=np.uint32)
    query = np.random.randint(class_count, size=query_size, dtype=np.uint32)

    hashes_class = np.random.rand(class_count, hash_size).astype(np.float32) - 0.5

    hashes_db = hashes_class[db] + 0.3 * np.random.randn(db_size, hash_size).astype(np.float32)
    hashes_query = hashes_class[query] + 0.3 * np.random.randn(query_size, hash_size).astype(np.float32)

    mAP, p, r = compute_map(hashes_db, hashes_query, db, query, query_size // 2)

    print(mAP, p, r)

    _mAP, p, r = compute_map_fast(hashes_db, hashes_query, db, query, query_size // 2)

    print(mAP, p, r)

    print("Passed!" if (_mAP == mAP) else "Failed!")

    _mAP, p, r = compute_map_faster(hashes_db, hashes_query, db, query, query_size // 2)

    print(mAP, p, r)

    print("Passed!" if (_mAP == mAP) else "Failed!")
