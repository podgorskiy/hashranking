from hashranking import hashranking_cpp


def hamming_rank(b1, b2):
    """Return rank of pairs. Takes vector of hashes b1 and b2 and returns correspondence rank of b1 to b2
    """
    dist_h = hashranking_cpp.hamming_distance(b1, b2)
    return hashranking_cpp.argsort(dist_h)
