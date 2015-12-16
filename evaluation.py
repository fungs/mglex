u"""
Submodule with all evaluation-related code.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import common
import itertools
import numpy as np
import sys


def chunkify(iterable, n):
    iterable = iter(iterable)
    n_rest = n - 1

    for item in iterable:
        rest = itertools.islice(iterable, n_rest)
        yield itertools.chain((item,), rest)


def pairs_nonzero(mat):
    caller_count = 0
    for moving_index in range(1, mat.shape[0]):
        caller_count += 1
        for truth_val in np.any(np.logical_and(mat[:-moving_index], mat[moving_index:]), axis=1):
            yield truth_val


def pairs_nonzero2(mat):
    left = []
    right = []
    n = mat.shape[0]
    size = 0
    for moving_index in range(1, mat.shape[0]):
        left.append(mat[:-moving_index])
        right.append(mat[moving_index:])
        size += n - moving_index
        if size >= n:
            for truth_val in np.any(np.logical_and(np.vstack(left), np.vstack(right)), axis=1):
                yield truth_val
            del left[:]
            del right[:]
            size = 0


def pairs(n):
    for s in range(1, n):
        for i, j in zip(range(n), range(s, n)):
            yield i, j


# TODO: incorporate weights, filter possible pairs before matrix arithmetics
def expected_pairwise_clustering_nonsparse(lmat, pmat, weights=None, subsample=None, blocksize=None, compress=False):
    assert lmat.shape == pmat.shape

    n, c = lmat.shape

    # default blocksize set to input matrix size
    if not blocksize:
        blocksize = n

    # random subsampling, if requested
    if subsample and subsample < n:
        rows_sampling = np.random.choice(n, subsample, replace=False)
        lmat = lmat[rows_sampling]
        pmat = pmat[rows_sampling]
        n = subsample

    # compress if requested
    if compress:
        mask = lmat.sum(dtype=np.bool_, axis=1)
        if not np.all(mask):
            lmat = lmat.compress(mask, axis=0)
            pmat = pmat.compress(mask, axis=0)
            n = lmat.shape[0]


    prob_sum = np.zeros(c, dtype=common.large_float_type)
    norm_term = np.zeros(c, dtype=common.large_float_type)

    indices = itertools.combinations(range(n), 2)

    for index_block in chunkify(indices, blocksize):
        i1, i2 = zip(*index_block)

        lprob = lmat.take(i1, axis=0) * lmat.take(i2, axis=0)
        pprob = np.sum(pmat.take(i1, axis=0) * pmat.take(i2, axis=0), axis=1, keepdims=True, dtype=common.large_float_type)

        block_prob_sum = lprob * pprob

        prob_sum += np.sum(block_prob_sum, axis=0, dtype=common.large_float_type)
        norm_term += np.sum(lprob, axis=0, dtype=common.large_float_type)

    #print(common.pretty_probvector(prob_sum))
    #print(common.pretty_probvector(norm_term))

    probs = prob_sum/norm_term

    sys.stderr.write("%.2f\t%.2f\t%s\n" % (np.mean(probs), np.sum(probs**2), common.pretty_probvector(probs)))
    return probs


def expected_pairwise_clustering(lmat, pmat, weights=None, subsample=None, blocksize=None, compress=False):
    assert lmat.shape == pmat.shape

    n, c = lmat.shape

    # default blocksize set to input matrix size
    if not blocksize:
        blocksize = n

    # random subsampling, if requested
    if subsample and subsample < n:
        rows_sampling = np.random.choice(n, subsample, replace=False)
        lmat = lmat[rows_sampling]
        pmat = pmat[rows_sampling]
        n = subsample

    # compress if requested
    if compress:
        mask = lmat.sum(dtype=np.bool_, axis=1)
        if not np.all(mask):
            lmat = lmat.compress(mask, axis=0)
            pmat = pmat.compress(mask, axis=0)
            n = lmat.shape[0]

    prob_sum = np.zeros(c, dtype=common.large_float_type)
    norm_term = np.zeros(c, dtype=common.large_float_type)

    indices = pairs(n)
    predicates = pairs_nonzero(lmat)
    filtered_indices = (e for (e, p) in zip(indices, predicates) if p)

    for index_block in chunkify(filtered_indices, blocksize):
        i1, i2 = zip(*index_block)

        lprob = lmat.take(i1, axis=0) * lmat.take(i2, axis=0)
        pprob = np.sum(pmat.take(i1, axis=0) * pmat.take(i2, axis=0), axis=1, keepdims=True, dtype=common.large_float_type)

        block_prob_sum = lprob * pprob

        prob_sum += np.sum(block_prob_sum, axis=0, dtype=common.large_float_type)
        norm_term += np.sum(lprob, axis=0, dtype=common.large_float_type)

    #print(common.pretty_probvector(prob_sum))
    #print(common.pretty_probvector(norm_term))

    probs = prob_sum/norm_term

    sys.stderr.write("%.2f\t%.2f\t%s\n" % (np.mean(probs), np.sum(probs**2), common.pretty_probvector(probs)))
    return probs


if __name__ == "__main__":
    pass
