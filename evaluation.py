u"""
Submodule with all evaluation-related code.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import common
import itertools
import numpy as np
import sys


def chunkify(iterable, n):
    it = iter(iterable)
    while True:
        blockiter = itertools.islice(it, n)
        try:
            firstitem = next(blockiter)
            yield itertools.chain((firstitem,), blockiter)
        except StopIteration:
            break


def expected_pairwise_clustering_simple(lmat, pmat):
    u"""
    Takes a label matrix one-zero entries and probability class assignments and calculates an evaluation statistic
    S = log((1/C) * \sum_i=1_C (1/|C_i|*(|C_i|-1)) * \sum_{d_1, d_2 \element C_i, d_1 != d_2} p(d_1|C_i)*p(d_2|C_i))
    The expected (log-)probability that any two linked contigs (prior knowledge) are grouped together in a cluster.

    This measure can be generalized to pairs of sequences which should _not_ belong together in a cluster (between)
    and for fuzzy label distributions.
    """

    predictions_per_labelgroup = {}

    negative_inf = -float("inf")
    for lvec, pvec in zip(lmat, pmat):
        number_nonzero = lvec.size - np.sum(lvec == negative_inf)
        if not number_nonzero:
            continue
        assert number_nonzero == 1

        i = np.nonzero(lvec == 0.)[0][0]  # only one index can be one by definition

        try:
            predictions_per_labelgroup[i].append(pvec)
        except KeyError:
            predictions_per_labelgroup[i] = [pvec]

    probs = np.zeros(len(predictions_per_labelgroup))

    for i, rows in enumerate(predictions_per_labelgroup.values()):
        mat = np.vstack(rows)
        group_probs = []
        for v1, v2 in permutations(mat, 2):
            group_probs.append(np.exp(v1+v2).sum())
        mprob = np.mean(group_probs)
        probs[i] = mprob

    sys.stderr.write("%.2f\t%.2f\t%s\n" % (np.mean(probs), np.sum(probs**2), common.pretty_probvector(probs)))
    return probs


def expected_pairwise_clustering_iterative(lmat, pmat, weights=None, subsample=None):
    assert lmat.shape == pmat.shape

    n, c = lmat.shape

    if subsample and subsample < n:
        rows_sampling = np.random.choice(n, subsample, replace=False)
        lmat = lmat[rows_sampling]
        pmat = pmat[rows_sampling]
        n = subsample

    prob_sum = np.zeros(c, dtype=common.large_float_type)
    norm_term = np.zeros(c, dtype=common.large_float_type)


    for (lvec1, pvec1), (lvec2, pvec2) in combinations(zip(lmat, pmat), 2):  # TODO: can also use permutation of indices
        lprob = lvec1*lvec2

        if np.any(lprob):
            pprob = np.dot(pvec1, pvec2)

            prob_sum += lprob * pprob
            sys.stderr.write("prob_term: %s\n" % lprob*pprob)

            norm_term += lprob

    #print(common.pretty_probvector(prob_sum))
    #print(common.pretty_probvector(norm_term))

    probs = prob_sum/norm_term
    sys.stderr.write("%.2f\t%.2f\t%s\n" % (np.mean(probs), np.sum(probs**2), common.pretty_probvector(probs)))
    return probs


# TODO: incorporate weights, filter possible pairs before matrix arithmetics
def expected_pairwise_clustering(lmat, pmat, weights=None, subsample=None, blocksize=None):
    assert lmat.shape == pmat.shape

    n, c = lmat.shape

    if subsample and subsample < n:
        rows_sampling = np.random.choice(n, subsample, replace=False)
        lmat = lmat[rows_sampling]
        pmat = pmat[rows_sampling]
        n = subsample

    if not blocksize:
        blocksize = n

    prob_sum = np.zeros(c, dtype=common.large_float_type)
    norm_term = np.zeros(c, dtype=common.large_float_type)

    indices = itertools.combinations(range(n), 2)

    for index_block in chunkify(indices, blocksize):
        i1, i2 = map(list, zip(*index_block))

        lprob = lmat[i1]*lmat[i2]
        pprob = np.sum(pmat[i1]*pmat[i2], axis=1, keepdims=True, dtype=common.large_float_type)

        block_prob_sum = lprob * pprob

        prob_sum += np.sum(block_prob_sum, axis=0, dtype=common.large_float_type)
        norm_term += np.sum(lprob, axis=0, dtype=common.large_float_type)

    #print(common.pretty_probvector(prob_sum))
    #print(common.pretty_probvector(norm_term))

    probs = prob_sum/norm_term

    sys.stderr.write("%.2f\t%.2f\t%s\n" % (np.mean(probs), np.sum(probs**2), common.pretty_probvector(probs)))
    return probs


def rows2array():
    pass


if __name__ == "__main__":
    pass
