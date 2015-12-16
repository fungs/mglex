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

    error = prob_sum/norm_term

    wmean = np.mean(error)  # TODO: use weighted mean here and in the separation method? -> both and unify
    mse = np.sqrt(np.sum(error**2))
    sys.stderr.write("%f\t%f\t%s\n" % (wmean, mse, common.pretty_probvector(error)))
    return mse


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

    error = prob_sum/norm_term

    wmean = np.mean(error)  # TODO: use weighted mean here and in the separation method? -> both and unify
    mse = np.sqrt(np.sum(error**2))
    sys.stderr.write("%f\t%f\t%s\n" % (wmean, mse, common.pretty_probvector(error)))
    return mse


def twoclass_separation(lmat, pmat, weights):
    assert lmat.shape == pmat.shape

    c = lmat.shape[1]
    scores = np.zeros(c, dtype=common.large_float_type)
    sizes = np.zeros(c, dtype=common.large_float_type)
    for i in range(c):
        r = pmat[:, (i,)]
        wn = r * weights
        wn_sum = wn.sum()
        wn /= wn_sum
        wa = (1.0 - r) * weights
        wa /= wa.sum()
        l = lmat[:, i]
        scores[i] = twoclass_separation_onecolumn(l, wn, wa)
        sizes[i] = wn_sum

    classpriors = sizes/sizes.sum()
    wmean = np.sum(classpriors*scores)
    mse = np.sqrt(np.sum((scores**2)*classpriors))
    sys.stderr.write("%f\t%f\t%s\n" % (wmean, mse, common.pretty_probvector(scores)))
    return mse


def twoclass_separation_onecolumn(like, weights_null, weights_alt):
    # TODO: do cumulative arrays and array arithmetics
    like = -like
    order = np.argsort(like, axis=0)

    error = 0.0
    wn_cumulative = common.large_float_type(weights_null.sum())
    wa_cumulative = common.large_float_type(0.0)

    l_last = 0.0
    step_size = 0.0
    width = 0.0
    for l, wn, wa, step in zip(like[order], weights_null[order], weights_alt[order], itertools.count()):  # TODO: stop loop when wn_cumulative == 0.0
        step_size = l - l_last
        if step_size > 0.0:
            height = wn_cumulative * wa_cumulative
            box = height * width
            error += box
            width = step_size
            #print("Step %i: like=%.2f, wn_cum=%f, wa_cum=%f, height=%f, width=%f, box=%f, error=%f" % (step, l, wn_cumulative, wa_cumulative, height, width, box, error))

        wn_cumulative -= wn
        wa_cumulative += wa
        l_last = l

    common.assert_approx_equal(wa_cumulative, 1.0)
    height = wn_cumulative * wa_cumulative
    box = height * width
    error += box
    error /= l_last
    return error


if __name__ == "__main__":
    pass
