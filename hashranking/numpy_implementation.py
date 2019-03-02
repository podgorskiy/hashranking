# Copyright 2017-2019 Stanislav Pidhorskyi
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


def calc_hamming_dist(b1, b2):
    """Compute the hamming distance between every pair of data points represented in each row of b1 and b2"""
    p1 = np.sign(b1).astype(np.int8)
    p2 = np.sign(b2).astype(np.int8)

    r = p1.shape[1]
    d = (r - np.matmul(p1, np.transpose(p2))) // 2
    return d


def calc_hamming_rank(b1, b2):
    """Return rank of pairs. Takes vector of hashes b1 and b2 and returns correspondence rank of b1 to b2
    """
    dist_h = calc_hamming_dist(b1, b2)
    return np.argsort(dist_h, 1, kind='stable')


def compute_map(hashes_db, hashes_query, labels_db, labels_query, top_n=0):
    """Compute MAP for given set of hashes and labels"""
    rank = calc_hamming_rank(hashes_query, hashes_db)
    s = __compute_s(labels_db, labels_query)
    return calc_map(rank, s, top_n)


def __compute_s(labels_db, labels_query, and_mode=False):
    """Return similarity matrix between two label vectors
    The output is binary matrix of size n_train x n_test
    """
    if and_mode:
        return np.bitwise_and(labels_db, np.transpose(labels_query)).astype(dtype=np.bool)
    else:
        return np.equal(labels_db, labels_query[:, np.newaxis])


def calc_map(rank, s, top_n):
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

