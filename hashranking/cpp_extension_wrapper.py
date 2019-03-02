import _hashranking


def calc_hamming_dist(b1, b2):
    return _hashranking.calc_hamming_dist(b1, b2)


def argsort(dist_h):
    """Return rank of pairs. Takes vector of hashes b1 and b2 and returns correspondence rank of b1 to b2
    """
    return _hashranking.argsort(dist_h)


def calc_hamming_rank(b1, b2):
    """Return rank of pairs. Takes vector of hashes b1 and b2 and returns correspondence rank of b1 to b2
    """
    dist_h = calc_hamming_dist(b1, b2)
    return _hashranking.argsort(dist_h)


def compute_map(hashes_db, hashes_query, labels_db, labels_query, top_n=0):
    """Compute MAP for given set of hashes and labels"""
    return _hashranking.calc_map_from_hashes(hashes_db, hashes_query,  labels_db, labels_query, top_n)


def compute_map_from_rank(rank, labels_db, labels_query, top_n=0):
    """Compute MAP for given set of hashes and labels"""
    return _hashranking.calc_map(rank,  labels_db, labels_query, top_n)
