hashranking - fast procedures for forking with hashes for deeplearning research
===============================================================================

In Deep Learning research, hashing, retrieval, and ranking tasks often require calculation of mAP of retrieval, which can be computationally expensive.
Often, hashes are represented as ndarrays of floats, where the sign of the float number has the meaning of a bit. This happens because neural networks work with floats and that is precisely the output that they give. Computing ranking of documents in the DB based the query typically implemented in NumPy, which can be quite suboptimal.

This package provides several functions that allow fast hamming distance computation, ranking, and mAP computation.

All API has two backends:
* NumPy implementation, which is simple, straightforward and as efficient as it can get in pure NumPy. Used as a reference.
* C++ Python extension that implements the same API, on average 10x faster than NumPy implementation and significantly more memory efficient.
