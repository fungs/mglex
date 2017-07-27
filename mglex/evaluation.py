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

    if compress: # remove all sequences with zero prob
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


def kbl_similarity(log_col1, log_col2, log_weight=None):
    "A probabilistic distance based on pairwise mixture likelihoods"

    tmp_pair = np.column_stack((log_col1, log_col2))  # creates a copy
    #common.write_probmatrix(tmp_pair, file=sys.stderr)

    # shift values before exp to avoid tiny numbers
    log_shift = tmp_pair.max(axis=1, keepdims=True)  # TODO: change name to log_factor
    
    with np.errstate(invalid='ignore'):
        tmp_pair -= log_shift  # we add this factor again, later
    
    # incorporate weights
    if log_weight is not None:  # weigh the summands
        log_shift += log_weight
    
    # reduce shift value by common factor and keep factors
    log_shift -= np.nanmax(log_shift)  # remove constant factor

    factor = np.squeeze(np.exp(log_shift, out=log_shift), axis=1)  # overwrites log_shift

    # compress data with factor of zero
    #assert not np.any(np.isnan(factor))  # there shouldn't be nans TODO: check this
    mask_nonzero = np.array(factor, dtype=np.bool_)
    number_nonzero = np.sum(mask_nonzero)
    if 2*number_nonzero > factor.size:  # hardcoded: compress arrays if >= 50% are zeroes
        #sys.stderr.write("Reducing number of entries in data to %i\n" % number_nonzero)
        tmp_pair[:number_nonzero] = tmp_pair[mask_nonzero]
        tmp_pair.resize((number_nonzero, 2))
        factor = factor[mask_nonzero]
 
    # calculate likelihood ratios for all data points
    log_col1 = tmp_pair[:, 0]  # first column view
    log_col2 = tmp_pair[:, 1]  # second column view
    log_sim = np.subtract(log_col2, log_col1, dtype=types.large_float_type)  # = log(p2/p1)
    #assert not np.any(np.isnan(log_sim))  # there can be nans due to (inf - inf)

    np.exp(tmp_pair, out=tmp_pair)
    col1 = log_col1  # reference
    col2 = log_col2  # reference

    with np.errstate(over='ignore'):
        ratio1 = np.exp(-log_sim)  # = p1/p2
        ratio2 = np.exp(log_sim)

    # workaround inf values
    mask = np.isinf(ratio1)
    if np.any(mask):
        log_sim[mask] = -log_sim[mask]  # p1/p2  == inf => p2 <<< p1 => log(p1/p2 + p2/p1) ~= log(p1/p2)

    # select elements which are finite in both ratios
    np.logical_not(mask, out=mask)
    np.logical_and(mask, np.isfinite(ratio2), out=mask)

    # calculate mixture likelihood for finite components
    log_sim[mask] = np.log(ratio2[mask] + 1./ratio2[mask])  # p1/p2 + p2/p1 = (p1^2 + p2^2)/(p1*p2)
    log_sim = np.subtract(np.log(2.), log_sim, out=log_sim)  # = 2*p1*p2/(p1^2+p2^2)
    #common.print_probvector(log_sim, file=sys.stderr)

    with np.errstate(invalid='ignore'):
        # (p1^2 + p2^2)/(p1 + p2) = (p1 + p2*(p2/p1))/(p2/p1) + 1) with p1 != 0
        # put large value in p1, small value in p2 (formula is symmetric)
        p1 = np.maximum(col1, col2)
        p2 = np.minimum(col1, col2)
        r2 = np.minimum(ratio1, ratio2)
        mix = np.divide(p1 + p2*r2, r2 + 1.)
        #mask = np.logical_and(col1, col2)
        #mix = np.ones(tmp_pair.shape[0], dtype=tmp_pair.dtype)
        #mix[mask] = np.divide(col1[mask] + col2[mask]*ratio2[mask], ratio2[mask] + 1.)
    #mix = (col1 ** 2 + col2 ** 2) / (col1 + col2)

    #common.print_probvector(mix, file=sys.stderr)

    np.multiply(mix, factor, out=mix)

    mix_sum = np.nansum(mix)
    
    if mix_sum == 0.:
        return np.nan
    
    with np.errstate(invalid='ignore'):
        log_sim *= mix
    
    return np.nansum(log_sim)/mix_sum
    

def combine_weight(log_weight, log_responsibility, i, j):
    """Add log columns in probability weights and combine with sequence length weights"""
    if log_responsibility is None:
        return log_weight
    #ret = np.maximum(log_responsibility[:, i], log_responsibility[:, j])[:, np.newaxis]  # only works for 1/0 weights
    
    # calculate log of sum of weights (needs 3n memory)
    data = log_responsibility[:, (i,j)]  # creates copy of columns
    data_col = data[:, 0]  # set reduced dim alias
    extra = np.max(data, axis=1, keepdims=True)
    extra_col = extra[:, 0]  # set reduced dim alias
    
    with np.errstate(invalid='ignore'):
        data -= extra  # shift: gives nan if both weights are zero (-inf + inf)
        
    np.exp(data, out=data)  # transform
    np.nansum(data, axis=1, out=data_col)  # sumup: set nans to zero
    np.log(data_col, out=data_col)  # transform back
    
    extra_col += data_col  # shift back
    
    if log_weight is None:
        return extra
    
    extra += log_weight
    return extra


def similarity_matrix(log_mat, log_weight=None, log_responsibility=None):
    """Calculate n*(n-1)/2 bin similarities by formula (2*L_b*L_b)/(L_a^2+L_b^2)"""

    n, d = log_mat.shape
    smat = np.zeros(shape=(d, d), dtype=types.logprob_type)  # TODO: use numpy triangle matrix object?
    
    if log_weight is not None:
        assert log_weight.shape[0] == n
        assert np.any(np.isfinite(log_weight))  # TODO: allow also zero weights
    
    for i in range(d):
        for j in range(i+1, d):
            # print("\n\ncol %i vs. %i:" % (i,j), file=sys.stderr)
            lw = combine_weight(log_weight, log_responsibility, i, j)
            #if lw is not None:
            #    sys.stderr.write("Nonzero elements in weights: %i\n" % sum(np.isfinite(lw)))
            log_p = kbl_similarity(log_mat[:, i], log_mat[:, j], lw)

            if log_p >= .0:
                smat[i, j] = smat[j, i] = 0.0
                if log_p > .0:
                    warnings.warn("Similarity larger than 1.0", UserWarning)
            else:
                smat[i, j] = smat[j, i] = log_p
    return smat


if __name__ == "__main__":
    pass
