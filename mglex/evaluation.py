u"""
Submodule with all evaluation-related code.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

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


def bin_distance_pairwise(col1, col2, weights, weights_sum=None, col1_sum=None, col2_sum=None):
    if col1_sum is None:
        col1_sum = np.sum(col1)

    if col2_sum is None:
        col2_sum = np.sum(col2)

    if weights_sum is None:
        weights_sum = np.sum(weights)

    component_wise_naive = np.divide((4*col1*col2), (col1+col2)**2)
    likelihood_weights_naive = (col1+col2)/(col1_sum+col2_sum)
    order = np.argsort(likelihood_weights_naive, axis=0)
    print(component_wise_naive[order][-10:-1], likelihood_weights_naive[order][-10:-1])
    return np.sum(np.multiply(component_wise_naive, likelihood_weights_naive))


    normterm = col1_sum+col2_sum
    factor = 4.0/normterm
    # factor = 4.0/(col1_sum+col2_sum)
    # factor = 4.0/(np.sum(col1)+np.sum(col2))
    print(col1.shape, col2.shape, weights.shape)
    nominator = 4*np.multiply(col1, col2)
    denominator = col1 + col2  # special case when both are zero!

    component_wise = (nominator/denominator)
    component_sum = np.sum(component_wise, dtype=types.large_float_type)

    print(factor.shape, nominator.shape, denominator.shape)
    print(factor, nominator[0], denominator[0], component_wise[0], file=sys.stderr)

    print(component_sum, normterm)
    return component_sum/normterm

    # return 4.0*np.sum((col1*col2)/(col1+col2))/(col1_sum+col2_sum)
    return factor * np.sum(nominator/denominator)


def S1(col1, col2):
    return np.divide(4*np.multiply(col1, col2), (col1+col2)**2)

def S2n(col1, col2):
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.vstack((np.divide(col1, col2), np.divide(col2, col1)))
        ratiosum = np.nansum(ratios, axis=0)
        assert np.all(~np.isnan(ratiosum))
        # print("# NaN:", np.sum(np.isnan(ratiosum)), file=sys.stderr)
        # print("# Inf:", np.sum(np.isinf(ratiosum)), file=sys.stderr)
        np.divide(2.0, ratiosum, out=ratiosum)
    return ratiosum

def S2n_log(col1, col2):
    with np.errstate(over='ignore'):
        ratios = np.exp(np.vstack((col1-col2, col2-col1)), dtype=types.large_float_type)
    ratiosum = np.sum(ratios, axis=0, dtype=types.large_float_type)
    assert np.all(~np.isnan(ratiosum))
    # print("# NaN:", np.sum(np.isnan(ratiosum)), file=sys.stderr)
    # print("# Inf:", np.sum(np.isinf(ratiosum)), file=sys.stderr)
    np.divide(2.0, ratiosum, out=ratiosum)
    return ratiosum

def S2(col1, col2):
    return np.divide(2*np.multiply(col1, col2), (col1**2+col2**2))

def S4(col1, col2):
    return np.sqrt(S1(col1, col2))

def similarity_diagnostics(col1, col2, fn):
    components = fn(col1, col2)
    col1_sum, col2_sum = col1.sum(dtype=types.large_float_type), col2.sum(dtype=types.large_float_type)
    lweights = np.divide(col1+col2, col1_sum+col2_sum)
    # order = np.argsort(lweights, axis=0)
    # print("components:", components[order][-2:-1], "weights:", lweights[order][-2:-1])
    return np.nansum(np.multiply(lweights, components), dtype=types.large_float_type)

def log_similarity_diagnostics(col1, col2, fn):
    components = fn(col1, col2)
    col1 = np.exp(col1, dtype=types.large_float_type)  # TODO: use exp_normalize
    col2 = np.exp(col2, dtype=types.large_float_type)  # TODO: use exp_normalize
    col1_sum, col2_sum = col1.sum(), col2.sum()
    lweights = np.divide(col1+col2, col1_sum+col2_sum)
    # order = np.argsort(lweights, axis=0)
    # print("components:", components[order][-2:-1], "weights:", lweights[order][-2:-1])
    return np.nansum(np.multiply(lweights, components), dtype=types.large_float_type)


def similarity_matrix(logmat, weights=None):  # TODO: implement sequence length weights and weights=None?
    """Calculate n*(n-1)/2 bin similarities by formula (2*L_b*L_b)/(L_a^2+L_b^2)"""

    mat = np.exp(logmat, dtype=types.large_float_type)
    n = mat.shape[1]
    smat = np.zeros(shape=(n, n), dtype=types.logprob_type)  # TODO: use numpy triangle matrix object?
    lsums = np.sum(mat, axis=0)
    # wsum = np.sum(weights, dtype=types.large_float_type)
    # w2 = np.divide(weights, wsum).ravel()

    for i in range(n):
        for j in range(i+1, n):
            col1 = mat.take(i, axis=1)  # np.exp(log_col1, dtype=types.large_float_type)  # TODO: use exp_normalize
            col2 = mat.take(j, axis=1)  # np.exp(log_col2, dtype=types.large_float_type)  # TODO: use exp_normalize
            log_col1 = logmat.take(i, axis=1)
            log_col2 = logmat.take(j, axis=1)
            c = S2n_log(log_col1, log_col2)
            # c = S2n(col1, col2)  # numerically less stable
            w1 = np.divide(col1+col2, lsums[i]+lsums[j])
            with np.errstate(invalid='ignore'):
                p = np.nansum(np.multiply(w1, c))
            #print("%i vs. %i: %.2f" % (i, j, p), file=sys.stderr)
            if p > 1.0:
                warnings.warn("Similarity larger than 1.0", UserWarning)
                cskew = np.any(c > 1.0)
                if cskew:
                    warnings.warn("Component similarity larger than 1.0", UserWarning)

            if p > 0:
                if p >= 1.0:
                    smat[i, j] = smat[j, i] = 0.0
                else:
                    smat[i, j] = smat[j, i] = np.log(p)  # TODO: better call log on entire results matrix
    return smat


if __name__ == "__main__":
    pass
