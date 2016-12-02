# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
Submodule with all evaluation-related code.
"""

__author__ = "code@fungs.de"

from . import common, types
import itertools
import numpy as np
import warnings
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
def expected_pairwise_clustering_nonsparse(lmat, pmat, weights=None, logarithmic=True, blocksize=None, compress=False):  # TODO: implement weights
    assert lmat.shape == pmat.shape

    if logarithmic:
        lmat = np.exp(lmat)
        pmat = np.exp(pmat)

    n, c = lmat.shape

    # default blocksize set to input matrix size
    if not blocksize:
        blocksize = n

    # compress if requested
    if compress:
        mask = lmat.sum(dtype=np.bool_, axis=1)
        if not np.all(mask):
            lmat = lmat.compress(mask, axis=0)
            pmat = pmat.compress(mask, axis=0)
            n = lmat.shape[0]


    prob_sum = np.zeros(c, dtype=types.large_float_type)
    norm_term = np.zeros(c, dtype=types.large_float_type)

    indices = itertools.combinations(range(n), 2)

    for index_block in chunkify(indices, blocksize):
        i1, i2 = zip(*index_block)

        lprob = lmat.take(i1, axis=0) * lmat.take(i2, axis=0)
        pprob = np.sum(pmat.take(i1, axis=0) * pmat.take(i2, axis=0), axis=1, keepdims=True, dtype=types.large_float_type)

        block_prob_sum = lprob * pprob

        prob_sum += np.sum(block_prob_sum, axis=0, dtype=types.large_float_type)
        norm_term += np.sum(lprob, axis=0, dtype=types.large_float_type)

    error = prob_sum/norm_term

    mpc = np.mean(error)  # TODO: use weighted mean here and in the separation method? -> both and unify
    # mse = np.sqrt(np.sum(error**2))
    # sys.stderr.write("%f\t%f\t%s\n" % (wmean, mse, common.pretty_probvector(error)))
    return mpc


def expected_pairwise_clustering(lmat, pmat, weights=None, logarithmic=True, blocksize=None, compress=False):  # TODO: implement weights
    assert lmat.shape == pmat.shape

    if logarithmic:
        lmat = np.exp(lmat)
        pmat = np.exp(pmat)

    n, c = lmat.shape

    # default blocksize set to input matrix size
    if not blocksize:
        blocksize = n

    # compress if requested
    if compress:
        mask = lmat.sum(dtype=np.bool_, axis=1)
        if not np.all(mask):
            lmat = lmat.compress(mask, axis=0)
            pmat = pmat.compress(mask, axis=0)
            n = lmat.shape[0]

    prob_sum = np.zeros(c, dtype=types.large_float_type)
    norm_term = np.zeros(c, dtype=types.large_float_type)

    indices = pairs(n)
    predicates = pairs_nonzero(lmat)
    filtered_indices = (e for (e, p) in zip(indices, predicates) if p)

    for index_block in chunkify(filtered_indices, blocksize):
        i1, i2 = zip(*index_block)

        lprob = lmat.take(i1, axis=0) * lmat.take(i2, axis=0)
        pprob = np.sum(pmat.take(i1, axis=0) * pmat.take(i2, axis=0), axis=1, keepdims=True, dtype=types.large_float_type)

        block_prob_sum = lprob * pprob

        prob_sum += np.sum(block_prob_sum, axis=0, dtype=types.large_float_type)
        norm_term += np.sum(lprob, axis=0, dtype=types.large_float_type)

    with np.errstate(invalid='ignore'):
        error = prob_sum/norm_term

    mpc = np.nanmean(error)  # TODO: use weighted mean here and in the separation method? -> both and unify
    # mse = np.sqrt(np.nansum(error**2))
    # sys.stderr.write("%f\t%f\t%s\n" % (wmean, mse, common.pretty_probvector(error)))
    return mpc


def twoclass_separation(lmat, pmat, weights=None, logarithmic=True):  # TODO: vectorize   # TODO: implement weights==None?
    assert lmat.shape == pmat.shape

    if not logarithmic:
        lmat = np.log(lmat)
        pmat = np.log(pmat)

    c = lmat.shape[1]
    scores = np.zeros(c, dtype=types.large_float_type)
    sizes = np.zeros(c, dtype=types.large_float_type)
    for i in range(c):
        r = np.exp(pmat[:, (i,)])
        wn = r * weights
        sizes[i] = wn_sum = wn.sum(dtype=types.large_float_type)
        if not wn_sum:  # skip unnecessary computations and invalid warnings
            scores[i] = np.nan
            continue
        wn /= wn_sum
        wa = (1.0 - r) * weights
        wa /= wa.sum(dtype=types.large_float_type)
        l = lmat[:, i]
        scores[i] = twoclass_separation_onecolumn(l, wn, wa)

    classpriors = sizes/sizes.sum()
    wmean = np.nansum(classpriors*scores)
    mse = np.sqrt(np.nansum((scores**2)*classpriors))
    sys.stderr.write("%f\t%f\t%s\n" % (wmean, mse, common.pretty_probvector(scores)))
    return mse


def twoclass_separation_onecolumn(like, weights_null, weights_alt):
    # TODO: do cumulative arrays and array arithmetics
    m = ~np.isnan(like)
    like = -like[m]
    order = np.argsort(like, axis=0)

    error = 0.0
    wn_cumulative = types.large_float_type(weights_null.sum())
    wa_cumulative = types.large_float_type(0.0)

    l_last = 0.0
    width = 0.0
    for l, wn, wa, step in zip(np.nan_to_num(like[order]), weights_null[m][order], weights_alt[m][order], itertools.count()):  # TODO: stop loop when wn_cumulative == 0.0
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


def mean_squarred_error(lmat, pmat, weights=None, logarithmic=True):  # TODO: implement weights==None?
    """Square-rooted mean squared error as a fast evaluation score"""

    if logarithmic:
        lmat = np.exp(lmat)
        pmat = np.exp(pmat)

    assert lmat.shape == pmat.shape, "Shape mismatch in prediction and truth matrix."
    mse = np.sum(np.sum((lmat - pmat)**2, axis=1, keepdims=True)*weights, dtype=types.large_float_type)
    return np.sqrt(mse/np.sum(weights)/4.0)


def kbl_similarity(log_col1, log_col2):
    tmp_pair = np.column_stack((log_col1, log_col2))  # creates a copy
    log_shift = tmp_pair.max(axis=1, keepdims=False)  # shift values before exp to avoid tiny numbers
    tmp_pair -= log_shift[:, np.newaxis]

    # reduce shift value by common factor and compress data factors are sparse
    log_shift -= log_shift.max()
    factor = np.exp(log_shift, out=log_shift)  # overwrites log_shift

    mask = np.array(factor, dtype=bool)
    number_nonzero = mask.sum()
    if 2*number_nonzero > factor.size:  # hard-coded threshold
        tmp_pair[:number_nonzero] = tmp_pair[mask]
        tmp_pair.resize((number_nonzero, 2))
        factor = factor[mask]

    log_col1 = tmp_pair[:, 0]  # first column view
    log_col2 = tmp_pair[:, 1]  # second column view
    log_sim = np.subtract(log_col2, log_col1, dtype=types.large_float_type)

    np.exp(tmp_pair, out=tmp_pair)
    col1 = log_col1  # reference
    col2 = log_col2  # reference

    with np.errstate(over='ignore'):
        ratio1 = np.exp(-log_sim)
        ratio2 = np.exp(log_sim)

    # workaround inf values
    mask = np.isinf(ratio1)
    # print("number of inf entries in ratio1:", sum(mask), file=sys.stderr)
    if np.any(mask):
        log_sim[mask] = -log_sim[mask]
    mask = np.negative(mask, out=mask)
    mask = np.logical_and(mask, np.isfinite(ratio2), out=mask)

    log_sim[mask] = np.log(ratio2[mask] + 1./ratio2[mask])
    log_sim = np.subtract(np.log(2.), log_sim, out=log_sim)
    assert np.all(np.isfinite(log_sim))

    with np.errstate(invalid='ignore'):
        mix = np.divide(col1 + col2*ratio2, ratio2 + 1.)

    np.multiply(mix, factor, out=mix)
    mix_sum = np.nansum(mix)
    # print("mix sum:", mix_sum, file=sys.stderr)
    assert mix_sum
    np.divide(mix, mix_sum, out=mix)
    with np.errstate(invalid='ignore'):
        log_sim *= mix
    return log_sim


def similarity_matrix(logmat, weights=None):  # TODO: implement sequence length weights and weights=None?
    """Calculate n*(n-1)/2 bin similarities by formula (2*L_b*L_b)/(L_a^2+L_b^2)"""

    n, d = logmat.shape
    smat = np.zeros(shape=(d, d), dtype=types.logprob_type)  # TODO: use numpy triangle matrix object?
    # lsums = np.sum(mat, axis=0)
    # wsum = np.sum(weights, dtype=types.large_float_type)
    # w2 = np.divide(weights, wsum).ravel()

    for i in range(d):
        for j in range(i+1, d):
            # print("\n\ncol %i vs. %i:" % (i,j), file=sys.stderr)
            log_p = np.nansum(kbl_similarity(logmat[:, i], logmat[:, j]))  # TODO: pass array instead
            # print("similarity value is:", p, file=sys.stderr)

            if log_p >= .0:
                smat[i, j] = smat[j, i] = 0.0
                if log_p > .0:
                    warnings.warn("Similarity larger than 1.0", UserWarning)
            else:
                smat[i, j] = smat[j, i] = log_p
    return smat


if __name__ == "__main__":
    pass
