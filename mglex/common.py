# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
This file contains helper functions and types.
"""

__author__ = "code@fungs.de"

from . import types
import numpy as np
import math
from numpy.testing import assert_approx_equal
from scipy.special import binom
from operator import itemgetter
from itertools import count, filterfalse, chain
from collections import defaultdict, deque
from sys import stderr, stdout
import pickle
from scipy.special import gammaln  # TODO: clear dependency on scipy


def parse_lines(lines):
    for line in lines:
        if not line or line[0] == "#":  # skip empty lines and comments
            continue
        yield line.rstrip()


load_data = lambda lines, store: store.parse(parse_lines(lines))
load_data_file = lambda filename, store: load_data(open(filename, "r"), store)


def assert_probmatrix(mat):
    is_sum = mat.sum(dtype=np.float32)
    should_sum = mat.shape[0]
    assert_approx_equal(is_sum, should_sum, significant=0)
    [assert_approx_equal(rowsum, 1., significant=1) for rowsum in mat.sum(axis=1, dtype=np.float32)]


def assert_probmatrix_relaxed(mat):  # accepts matrices with all-nan rows (invalid training data for class etc.)
    mask = ~np.all(np.isnan(mat), axis=1, keepdims=False)
    mat = mat.compress(mask, axis=0)
    assert_probmatrix(mat)


def approx_equal(v1, v2, precision):
    if type(v1) == type(v2) == np.ndarray:
        if v1.shape != v2.shape:
            return False
        return (abs(v1-v2) < precision).all()
    return abs(v1-v2) < precision


assert_probarray = lambda v: assert_approx_equal(v.sum(), 1.)


def random_probarray(size):  # TODO: refine
    tmp = np.random.rand(size)
    return tmp/tmp.sum()


def set_random_seed(seed):
    np.random.seed(seed)


def argmax(s, n=1):
    get_second = itemgetter(1)
    max_store = sorted(list(enumerate(s[:n])), key=get_second, reverse=True)
    for e in zip(count(n), s[n:]):
        max_store = sorted(max_store + [e], key=get_second, reverse=True)[:n]
    if n == 1:
        return max_store[0]
    return max_store


def logbinom(n, k):
    return gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)


def logmultinom(n, k):
    return gammaln(n+1) - np.sum(gammaln(k+1), axis=1, keepdims=True)


def nandot(a, b):  # TODO: speed up, avoid copying data
    "A numpy.dot() replacement which treats (0*-Inf)==0 and works around BLAS NaN bugs in matrices."
    # important note: a contains zeros and b contains inf/-inf/nan, not the other way around

    # workaround for zero*-inf=nan in dot product (must be 0 according to 0^0=1 with probabilities)
    # 1) calculate dot product
    # 2) select nan entries
    # 3) re-calculate matrix entries where 0*inf = 0 using np.nansum()
    tmp = np.dot(a, b)
    indices = np.where(np.isnan(tmp))
    ri, ci = indices
    with np.errstate(invalid='ignore'):
        values = np.nansum(a[ri, :] * b[:, ci].T, axis=1)
    values[np.isnan(values)] = 0.0
    tmp[indices] = values
    return tmp


flat_priors = lambda n: np.repeat(1./n, n)


def total_likelihood_inplace(log_mat):
    correction = np.max(log_mat, axis=1, keepdims=True)  # tiny number correction
    log_mat -= correction
    l_per_datum = np.exp(log_mat).sum(axis=1)
    log_vec = np.log(l_per_datum) + correction
    return log_vec.sum()


def total_likelihood(log_mat):
    correction = np.max(log_mat, axis=1, keepdims=True)  # tiny number correction
    tmp = log_mat - correction
    l_per_datum = np.exp(tmp).sum(axis=1)
    log_vec = np.log(l_per_datum) + correction
    return log_vec.sum()


def exp_normalize_inplace(data):  # important: works in-place
    data -= data.max(axis=1, keepdims=True)  # avoid tiny numbers
    data = np.exp(data)
    data /= data.asum(axis=1, keepdims=True)
    return data


def exp_normalize(data):
    ret = data - np.amax(data, axis=1, keepdims=True)  # avoid tiny numbers
    ret = np.exp(ret)
    ret /= np.sum(ret, axis=1, keepdims=True)
    return ret


def exp_normalize_1d_inplace(data):  # important: works in-place
    data -= data.max()  # avoid tiny numbers
    data = np.exp(data)
    data /= data.sum()
    return data


def exp_normalize_1d(data):
    ret = data - data.max()  # avoid tiny numbers
    ret = np.exp(ret)
    ret /= ret.sum()
    return ret


swapindex_2d = [1, 0]


def weighted_std_matrix(data, weights, dtype=types.large_float_type, shrink_matrix=True):  # TODO adjust return of NaN and zero
    """Weighted standard deviation using numpy masked arrays"""
    assert weights.shape == data.shape

    max_valid_value = np.floor(np.sqrt(np.finfo(data.dtype).max))
    # print("max value:", np.abs(data).max(), file=stderr)

    # shrink and copy original data
    data_weighted_var = np.zeros(data.shape[1], dtype=dtype)
    d = data
    w = weights
    m = ~np.logical_and(w, np.isfinite(d))

    if shrink_matrix:
        print(d.shape, w.shape, m.shape, file=stderr)
        select = ~np.all(m, axis=1)
        if np.any(select):
            d = np.compress(select, d, axis=0)
            w = np.compress(select, w, axis=0)
            m = np.compress(select, m, axis=0)
            print(d.shape, w.shape, m.shape, file=stderr)
        select = ~np.all(m, axis=0)
        if np.any(select):
            d = np.compress(select, d, axis=1)
            w = np.compress(select, w, axis=1)
            m = np.compress(select, m, axis=1)
            print(d.shape, w.shape, m.shape, file=stderr)
    else:
        d = d.copy()
        select = np.ones(data.shape[1], dtype=np.bool)

    assert d.shape == m.shape

    d = np.ma.MaskedArray(d, mask=m)
    w = np.ma.MaskedArray(w, mask=m)
    # print("max value:", np.ma.abs(d).max(fill_value=0.0), file=stderr)
    # d -= np.ma.mean(d, dtype=types.large_float_type)  # TODO: enable if overflow error in weighted mean calculation
    # print("max value:", np.ma.abs(d).max(fill_value=0.0), file=stderr)
    weight_sums = w.sum(dtype=types.large_float_type, axis=0)
    # d -= np.ma.average(np.ma.MaskedArray(d, dtype=types.large_float_type), weights=w, axis=0)  # TODO: avoid cast
    with np.errstate(invalid='ignore'):
        d -= np.ma.sum(d * w, axis=0)/weight_sums
    # print("max value:", np.ma.abs(d).max(fill_value=0.0), file=stderr)

    max_value = np.ma.abs(d).max(fill_value=0.0)
    if max_value > max_valid_value:
        shrink_divisor = max_value/(max_valid_value-1.0)
        # print("shrink divisor:", shrink_divisor, file=stderr)

        with np.errstate(invalid='ignore'):
            d /= shrink_divisor
        # print("max value after shrinking:", np.ma.abs(d).max(fill_value=0.0))
        # print_probvector(d[~m].flatten(), file=stderr)
        assert np.ma.abs(d).max() <= max_valid_value
    else:
        shrink_divisor = 1.0

    try:
        with np.errstate(over='raise'):
            d **= 2
    except FloatingPointError:
        stderr.write("Error: overflow in squared vector.\n")

    assert np.all(np.isfinite(d))

    variance_divisor = weight_sums - ((w**2).sum(dtype=dtype, axis=0)/weight_sums)  # replaces weight_sums in biased std
    with np.errstate(invalid='raise'):
        try:
            # data_weighted_var = shrink_divisor**2 * np.ma.average(np.ma.array(d, dtype=types.large_float_type), weights=w, axis=0)
            data_weighted_var[select] = np.ma.sqrt(np.ma.sum(d*w, axis=0)) * (shrink_divisor/np.sqrt(variance_divisor))
        except FloatingPointError:
            stderr.write("Error: probable overflow in np.average.\n")
            raise FloatingPointError

    assert np.all(data_weighted_var >= 0.0)
    return data_weighted_var


def weighted_std_iterative(data, weights, dtype=types.large_float_type):
    """Unbiased weighted standard deviation using iteration over columns, returns NaN if number of valid samples is < 2"""
    # unbiased version for reliabilty weights: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

    assert weights.shape == data.shape
    axis = 0
    original_dtype = data.dtype
    if dtype is None:
        dtype = original_dtype

    axis = swapindex_2d[axis]  # TODO: remove
    max_valid_value = np.floor(np.sqrt(np.finfo(data.dtype).max))
    data_weighted_var = np.empty(data.shape[axis], dtype=types.large_float_type)
    # data_weighted_var_mask = np.empty(data.shape[axis], dtype=np.bool)

    for i, d, w in zip(count(0), np.rollaxis(data, axis), np.rollaxis(weights, axis)):
        # ignore nan or infinity values
        m = np.isfinite(d)
        if sum(m) < 2:
            data_weighted_var[i] = np.nan
            # data_weighted_var_mask[i] = True
            continue

        np.logical_and(m, w, out=m)

        d = d[m]  # create memory copy
        if d.size < 2:
            data_weighted_var[i] = 0.0
            continue

        w = w[m]  # create memory copy

        weight_sum = w.sum(dtype=dtype)
        if not weight_sum:
            data_weighted_var[i] = 0.0
            continue

        # d -= np.mean(d, dtype=types.large_float_type)  # TODO: enable if overflow error in weighted mean calculation
        d -= (d*w).sum(dtype=dtype)/weight_sum

        max_value = np.abs(d).max()
        if max_value > max_valid_value:
            shrink_divisor = max_value/(max_valid_value-1.0)
            # print("shrink divisor:", shrink_divisor, file=stderr)
            d /= shrink_divisor
            # print("max value after shrinking:", np.abs(d).max())
            assert np.ma.abs(d).max() <= max_valid_value
        else:
            shrink_divisor = 1.0

        try:
            with np.errstate(over='raise'):
                # print("Min-Max square:", v.min(), v.max(), file=stderr)
                d **= 2
        except FloatingPointError:
            stderr.write("Error in weighted variance calculation: overflow in squared vector.\n")
            raise FloatingPointError

        variance_divisor = weight_sum - ((w**2).sum(dtype=dtype)/weight_sum)  # replaces weight_sums in biased std
        with np.errstate(over='raise'):
            try:
                data_weighted_var[i] = np.sqrt((d*w).sum(dtype=dtype)) * (shrink_divisor/np.sqrt(variance_divisor)) #np.average(np.array(d, dtype=types.large_float_type), weights=w)
            except FloatingPointError:
                stderr.write("Error in weighted variance calculation: probable overflow in weights*coverage calculation.\n")
                raise FloatingPointError
                # data_weighted_var[i] = np.inf

        assert data_weighted_var[i] >= 0.0
        # data_weighted_var_mask[i] = False

    # print_probvector(data_weighted_var, file=stderr)
    # return np.ma.MaskedArray(data_weighted_var, mask=data_weighted_var_mask)
    return data_weighted_var

weighted_std = weighted_std_iterative


def log_fac(i):
    r = .0
    while i > 0:
        r += np.exp(i)
        i -= 1
    return r


def seeds2indices(seqnames, seeds):  # TODO: deprecated -> remove
    # a) build a dictionary for the seeds for fast lookup
    name2cluster = {}
    cluster_count = 0
    for i, names in enumerate(seeds):
        for n in names:
            name2cluster[n] = i
        cluster_count += 1

    seed_indices = [set() for i in range(cluster_count)]

    # b) determine indices of seeds
    for i, name in enumerate(seqnames):
        cluster_index = name2cluster.get(name, None)
        if cluster_index is not None:
            seed_indices[cluster_index].add(i)
    return seed_indices


def responsibilities_from_seeds(seed_indices, num_data):  # TODO: deprecated -> remove
    responsibilities = np.zeros((num_data, len(seed_indices)), dtype=types.prob_type)
    for i, s in enumerate(seed_indices):
        responsibilities[list(s), i] = 1.  # TODO: index with numpy array instead of list?
    return responsibilities

def seeds2classindex(seeds):
    name2cluster = {}
    for i, names in enumerate(seeds):
        for n in names:
            name2cluster[n] = i
    return name2cluster

def seeds2responsibility_iter(seqnames, seeds):
    seeds = list(seeds)
    lookup = seeds2classindex(seeds)
    template = np.repeat(types.logprob_type('-inf'), len(seeds))
    for name in seqnames:
        index = lookup.get(name, None)
        row = template.copy()
        if index is not None:
            row[index] = 0.
        yield row


# def responsibilities_from_seeds(data, seeds):
#     num_clusters = len(seeds)
#     num_data = data.num_data
#
#     # a) build a dictionary for the seeds for fast lookup
#     name2cluster = {}
#     for i, names in enumerate(seeds):
#         for n in names:
#             name2cluster[n] = i
#
#     # b) construct zero-filled responsibility matrix
#     responsibilities = np.zeros((num_data, num_clusters), dtype=prob_type)
#
#     # c) fill in ones for seeds into responsibilities
#     for name, row in zip(data.names, responsibilities):
#         cluster_index = name2cluster.get(name, None)
#         if cluster_index is not None:
#             # print >>stderr, "assigning", name, "to cluster", cluster_index
#             row[cluster_index] = 1.
#
#     seqs_per_component = responsibilities.sum(axis=0)
#     print >>stderr, "number of contigs per component", seqs_per_component
#     assert(all(seqs_per_component))
#
#     return responsibilities


def load_seeds(iterable):
    for line in iterable:
        if line and line[0] == "#":  # TODO: factorize
            continue
        yield line.rstrip().split(" ")

load_seeds_file = lambda filename: load_seeds(open(filename, "r"))

load_seqlens_iter = lambda lines: (types.seqlen_type(line.rstrip()) for line in lines)
load_seqlens = lambda lines: np.fromiter(load_seqlens_iter(lines), dtype=types.seqlen_type)[:, np.newaxis]
load_seqlens_file = lambda filename: load_seqlens(open(filename, "r"))

load_seqnames_iter = lambda lines: (line.rstrip() for line in lines)
load_seqnames_file = lambda filename: load_seqnames_iter(open(filename, "r"))

load_model = pickle.load
load_model_file = lambda filename: load_model(open(filename, "rb"))

write_model = pickle.dump
write_model_file = lambda model, filename: write_model(model, open(filename, "wb"))

load_probmatrix_iter = lambda lines: (-np.array(line.split("\t"), dtype=types.logprob_type) for line in lines)
load_probmatrix = lambda lines: np.vstack(load_probmatrix_iter(lines))
load_probmatrix_file = lambda filename: load_probmatrix(open(filename, "r"))

def write_probmatrix_iter(rows, file=stdout):
    trans = lambda row: np.abs(np.asarray(row, dtype=types.logprob_type))  # assuming log-negative values
    for row in map(np.asarray, rows):
        file.write("\t".join(["%.2f" % i for i in trans(row)]))
        file.write("\n")

def write_probmatrix(mat, file=stdout):
    mat = np.abs(np.asarray(mat, dtype=types.logprob_type))  # assuming log-negative values
    for row in mat:
        file.write("\t".join(["%.2f" % i for i in row]))
        file.write("\n")

write_probmatrix_file = lambda mat, filename: write_probmatrix(mat, open(filename, "w"))

colors_dict = {
    'automatic'              : '#add8e6',     # 173, 216, 230
    'aliceblue'              : '#f0f8ff',     # 240, 248, 255
    'antiquewhite'           : '#faebd7',     # 250, 235, 215
    'aqua'                   : '#00ffff',     #   0, 255, 255
    'aquamarine'             : '#7fffd4',     # 127, 255, 212
    'azure'                  : '#f0ffff',     # 240, 255, 255
    'beige'                  : '#f5f5dc',     # 245, 245, 220
    'bisque'                 : '#ffe4c4',     # 255, 228, 196
    'black'                  : '#000000',     #   0,   0,   0
    'blanchedalmond'         : '#ffebcd',     # 255, 235, 205
    'blue'                   : '#0000ff',     #   0,   0, 255
    'blueviolet'             : '#8a2be2',     # 138,  43, 226
    'brown'                  : '#a52a2a',     # 165,  42,  42
    'burlywood'              : '#deb887',     # 222, 184, 135
    'cadetblue'              : '#5f9ea0',     #  95, 158, 160
    'chartreuse'             : '#7fff00',     # 127, 255,   0
    'chocolate'              : '#d2691e',     # 210, 105,  30
    'coral'                  : '#ff7f50',     # 255, 127,  80
    'cornflowerblue'         : '#6495ed',     # 100, 149, 237
    'cornsilk'               : '#fff8dc',     # 255, 248, 220
    'crimson'                : '#dc143c',     # 220,  20,  60
    'cyan'                   : '#00ffff',     #   0, 255, 255
    'darkblue'               : '#00008b',     #   0,   0, 139
    'darkcyan'               : '#008b8b',     #   0, 139, 139
    'darkgoldenrod'          : '#b8860b',     # 184, 134,  11
    'darkgray'               : '#a9a9a9',     # 169, 169, 169
    'darkgreen'              : '#006400',     #   0, 100,   0
    'darkgrey'               : '#a9a9a9',     # 169, 169, 169
    'darkkhaki'              : '#bdb76b',     # 189, 183, 107
    'darkmagenta'            : '#8b008b',     # 139,   0, 139
    'darkolivegreen'         : '#556b2f',     #  85, 107,  47
    'darkorange'             : '#ff8c00',     # 255, 140,   0
    'darkorchid'             : '#9932cc',     # 153,  50, 204
    'darkred'                : '#8b0000',     # 139,   0,   0
    'darksalmon'             : '#e9967a',     # 233, 150, 122
    'darkseagreen'           : '#8fbc8f',     # 143, 188, 143
    'darkslateblue'          : '#483d8b',     #  72,  61, 139
    'darkslategray'          : '#2f4f4f',     #  47,  79,  79
    'darkslategrey'          : '#2f4f4f',     #  47,  79,  79
    'darkturquoise'          : '#00ced1',     #   0, 206, 209
    'darkviolet'             : '#9400d3',     # 148,   0, 211
    'deeppink'               : '#ff1493',     # 255,  20, 147
    'deepskyblue'            : '#00bfff',     #   0, 191, 255
    'dimgray'                : '#696969',     # 105, 105, 105
    'dimgrey'                : '#696969',     # 105, 105, 105
    'dodgerblue'             : '#1e90ff',     #  30, 144, 255
    'firebrick'              : '#b22222',     # 178,  34,  34
    'floralwhite'            : '#fffaf0',     # 255, 250, 240
    'forestgreen'            : '#228b22',     #  34, 139,  34
    'fuchsia'                : '#ff00ff',     # 255,   0, 255
    'gainsboro'              : '#dcdcdc',     # 220, 220, 220
    'ghostwhite'             : '#f8f8ff',     # 248, 248, 255
    'gold'                   : '#ffd700',     # 255, 215,   0
    'goldenrod'              : '#daa520',     # 218, 165,  32
    'gray'                   : '#808080',     # 128, 128, 128
    'green'                  : '#008000',     #   0, 128,   0
    'greenyellow'            : '#adff2f',     # 173, 255,  47
    'grey'                   : '#808080',     # 128, 128, 128
    'honeydew'               : '#f0fff0',     # 240, 255, 240
    'hotpink'                : '#ff69b4',     # 255, 105, 180
    'indianred'              : '#cd5c5c',     # 205,  92,  92
    'indigo'                 : '#4b0082',     #  75,   0, 130
    'ivory'                  : '#fffff0',     # 255, 255, 240
    'khaki'                  : '#f0e68c',     # 240, 230, 140
    'lavender'               : '#e6e6fa',     # 230, 230, 250
    'lavenderblush'          : '#fff0f5',     # 255, 240, 245
    'lawngreen'              : '#7cfc00',     # 124, 252,   0
    'lemonchiffon'           : '#fffacd',     # 255, 250, 205
    'lightblue'              : '#add8e6',     # 173, 216, 230
    'lightcoral'             : '#f08080',     # 240, 128, 128
    'lightcyan'              : '#e0ffff',     # 224, 255, 255
    'lightgoldenrodyellow'   : '#fafad2',     # 250, 250, 210
    'lightgray'              : '#d3d3d3',     # 211, 211, 211
    'lightgreen'             : '#90ee90',     # 144, 238, 144
    'lightgrey'              : '#d3d3d3',     # 211, 211, 211
    'lightpink'              : '#ffb6c1',     # 255, 182, 193
    'lightsalmon'            : '#ffa07a',     # 255, 160, 122
    'lightseagreen'          : '#20b2aa',     #  32, 178, 170
    'lightskyblue'           : '#87cefa',     # 135, 206, 250
    'lightslategray'         : '#778899',     # 119, 136, 153
    'lightslategrey'         : '#778899',     # 119, 136, 153
    'lightsteelblue'         : '#b0c4de',     # 176, 196, 222
    'lightyellow'            : '#ffffe0',     # 255, 255, 224
    'lime'                   : '#00ff00',     #   0, 255,   0
    'limegreen'              : '#32cd32',     #  50, 205,  50
    'linen'                  : '#faf0e6',     # 250, 240, 230
    'magenta'                : '#ff00ff',     # 255,   0, 255
    'maroon'                 : '#800000',     # 128,   0,   0
    'mediumaquamarine'       : '#66cdaa',     # 102, 205, 170
    'mediumblue'             : '#0000cd',     #   0,   0, 205
    'mediumorchid'           : '#ba55d3',     # 186,  85, 211
    'mediumpurple'           : '#9370db',     # 147, 112, 219
    'mediumseagreen'         : '#3cb371',     #  60, 179, 113
    'mediumslateblue'        : '#7b68ee',     # 123, 104, 238
    'mediumspringgreen'      : '#00fa9a',     #   0, 250, 154
    'mediumturquoise'        : '#48d1cc',     #  72, 209, 204
    'mediumvioletred'        : '#c71585',     # 199,  21, 133
    'midnightblue'           : '#191970',     #  25,  25, 112
    'mintcream'              : '#f5fffa',     # 245, 255, 250
    'mistyrose'              : '#ffe4e1',     # 255, 228, 225
    'moccasin'               : '#ffe4b5',     # 255, 228, 181
    'navajowhite'            : '#ffdead',     # 255, 222, 173
    'navy'                   : '#000080',     #   0,   0, 128
    'oldlace'                : '#fdf5e6',     # 253, 245, 230
    'olive'                  : '#808000',     # 128, 128,   0
    'olivedrab'              : '#6b8e23',     # 107, 142,  35
    'orange'                 : '#ffa500',     # 255, 165,   0
    'orangered'              : '#ff4500',     # 255,  69,   0
    'orchid'                 : '#da70d6',     # 218, 112, 214
    'palegoldenrod'          : '#eee8aa',     # 238, 232, 170
    'palegreen'              : '#98fb98',     # 152, 251, 152
    'paleturquoise'          : '#afeeee',     # 175, 238, 238
    'palevioletred'          : '#db7093',     # 219, 112, 147
    'papayawhip'             : '#ffefd5',     # 255, 239, 213
    'peachpuff'              : '#ffdab9',     # 255, 218, 185
    'peru'                   : '#cd853f',     # 205, 133,  63
    'pink'                   : '#ffc0cb',     # 255, 192, 203
    'plum'                   : '#dda0dd',     # 221, 160, 221
    'powderblue'             : '#b0e0e6',     # 176, 224, 230
    'purple'                 : '#800080',     # 128,   0, 128
    'red'                    : '#ff0000',     # 255,   0,   0
    'rosybrown'              : '#bc8f8f',     # 188, 143, 143
    'royalblue'              : '#4169e1',     #  65, 105, 225
    'saddlebrown'            : '#8b4513',     # 139,  69,  19
    'salmon'                 : '#fa8072',     # 250, 128, 114
    'sandybrown'             : '#f4a460',     # 244, 164,  96
    'seagreen'               : '#2e8b57',     #  46, 139,  87
    'seashell'               : '#fff5ee',     # 255, 245, 238
    'sienna'                 : '#a0522d',     # 160,  82,  45
    'silver'                 : '#c0c0c0',     # 192, 192, 192
    'skyblue'                : '#87ceeb',     # 135, 206, 235
    'slateblue'              : '#6a5acd',     # 106,  90, 205
    'slategray'              : '#708090',     # 112, 128, 144
    'slategrey'              : '#708090',     # 112, 128, 144
    'snow'                   : '#fffafa',     # 255, 250, 250
    'springgreen'            : '#00ff7f',     #   0, 255, 127
    'steelblue'              : '#4682b4',     #  70, 130, 180
    'tan'                    : '#d2b48c',     # 210, 180, 140
    'teal'                   : '#008080',     #   0, 128, 128
    'thistle'                : '#d8bfd8',     # 216, 191, 216
    'tomato'                 : '#ff6347',     # 255,  99,  71
    'turquoise'              : '#40e0d0',     #  64, 224, 208
    'violet'                 : '#ee82ee',     # 238, 130, 238
    'wheat'                  : '#f5deb3',     # 245, 222, 179
    'white'                  : '#ffffff',     # 255, 255, 255
    'whitesmoke'             : '#f5f5f5',     # 245, 245, 245
    'yellow'                 : '#ffff00',     # 255, 255,   0
    'yellowgreen'            : '#9acd32'      # 154, 205,  50
}


def plot_clusters_pca(responsibilities, color_groups):
    from sklearn.decomposition import RandomizedPCA
    import pylab as pl
    from random import shuffle

    colors = list(colors_dict.values())
    shuffle(colors)

    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(responsibilities)
    # print >>stderr, pca.explained_variance_ratio_

    pl.figure()
    pl.scatter(X[:, 0], X[:, 1], c="grey", label="unknown")
    for c, sub, i in zip(colors, color_groups, count(0)):
        pl.scatter(X[sub, 0], X[sub, 1], c=c, label=str(i))
    pl.legend()
    pl.title("PCA responsibility matrix")
    pl.show()


hellinger_distance = lambda u, v: -np.log(1.001-np.sqrt(((np.sqrt(u) - np.sqrt(v))**2.).sum())/np.sqrt(2.))


my_distance = lambda u, v: -np.log(1.0001 - np.dot(u, v))


dummy_distance = lambda u, v: 0.


def plot_clusters_igraph(responsibilities, color_groups):
    from scipy.spatial.distance import pdist, correlation, squareform
    from igraph import Graph, plot
    data = responsibilities[:, :2]
    Y = pdist(data, hellinger_distance)
    print(Y[:30], file=stderr)
    # return
    g = Graph()
    n = data.shape[0]
    g.add_vertices(n)
    colors = ["grey"]*n
    palette = list(colors_dict.values())
    for j, group in enumerate(color_groups):
        c = palette[j]
        for i in group:
            colors[i] = c
    l = g.layout_mds(dist=squareform(Y))
    plot(g, layout=l, vertex_color=colors, bbox=(1024, 1024), vertex_size=5)


# c&p from stackexchange
def uniq(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element



def print_probmatrix(mat, file=stdout):
    for row in np.asarray(mat):
        file.write("\t".join(["%.2f" % i for i in row]))
        file.write("\n")



pretty_probvector = lambda vec: "|".join(("%.2f" % f for f in vec))
pretty_probmatrix = lambda mat: "\n".join((pretty_probvector(row) for row in mat))

def print_probvector(vec, file=stdout):
    file.write("|".join(("%.2f" % f for f in vec)))
    file.write("\n")


def print_vector(vec, file=stdout):
    file.write("|".join(("%s" % i for i in vec)))
    file.write("\n")


def newline(file=stdout):
    file.write("\n")


print_predictions = lambda mat: print_probmatrix(np.absolute(np.log(mat)))  # TODO: add proper file sink


# debug function
def factorial_array(vec):
    return np.asarray([np.math.factorial(i) for i in vec])  # superslow?


# debug function
def log_array(vec):
    return np.asarray([math.log(i) for i in vec], dtype=float)  # why not numpy.log?


binom_array = binom


class InternalTreeIndex:
    def __init__(self):
        self._store = defaultdict(self._context())

    def __getitem__(self, itemseq):
        current = self._store
        for item in itemseq:
            index, current = current[item]
            yield index

    def _context(self):
        obj = self._convert_generator_functor(count())
        return lambda: self._default_value(obj)

    def _default_value(self, obj):
        return obj(), defaultdict(self._context())

    def items(self):  # iterate breadth-first
        stack = deque([(tuple(), tuple(), self._store)])
        while stack:
            prefix_ext, prefix_int, store = stack.popleft()
            for node_ext, value in store.items():
                node_int, store_next = value
                path_ext = prefix_ext + (node_ext,)
                path_int = prefix_int + (node_int,)
                if store_next:
                    stack.append((path_ext, path_int, store_next))
                yield path_ext, path_int

    def keys(self):  # iterate breadth-first
        stack = deque([(tuple(), self._store)])
        while stack:
            prefix_ext, store = stack.popleft()
            for node_ext, value in store.items():
                store_next = value[1]
                path_ext = prefix_ext + (node_ext,)
                if store_next:
                    stack.append((path_ext, store_next))
                yield path_ext

    def values(self):  # iterate breadth-first
        stack = deque([(tuple(), self._store)])
        while stack:
            prefix_int, store = stack.popleft()[1:]
            for node_int, store_next in store.values():
                path_int = prefix_int + (node_int,)
                if store_next:
                    stack.append((path_int, store_next))
                yield path_int

    _convert_generator_functor = staticmethod(lambda gen: lambda: next(gen))


class NestedCountIndex:  # TODO: implement using NestedDict
    def __init__(self):
        self._store = self._fn()
        self._size = 0

    def _fn(self):
        return defaultdict(self._fn)

    def __getitem__(self, itemseq):
        current = self._store
        for item in itemseq:
            current = current[item]
        ret = current.get(self._defaultkey)
        if ret is None:
            ret = self._size
            current[self._defaultkey] = ret
            self._size += 1
        return ret

    def __len__(self):
        return self._size

    def keys(self):  # iterate breadth-first
        for path, val in self.items():
            yield path

    def _values_partial(self, queue):  # iterate breadth-first
        new = queue.popleft()
        while new is not None:  # invariant: level end
            store = new
            for node, val in store.items():
                if node is self._defaultkey:
                    yield val
                elif val:
                    queue.append(val)
            new = queue.popleft()
        raise StopIteration

    def values_nested(self):
        queue = deque([self._store])
        while queue:
            queue.append(None)  # invariant: level end
            yield self._values_partial(queue)

    def values(self):  # iterate breadth-first
        return chain.from_iterable(self.values_nested())

    def _items_partial(self, queue):  # iterate breadth-first
        new = queue.popleft()
        while new is not None:  # invariant: level end
            prefix, store = new
            for node, val in store.items():
                if node is self._defaultkey:
                    yield prefix, val
                elif val:
                    queue.append((prefix + (node,), val))
            new = queue.popleft()
        raise StopIteration

    def items_nested(self):
        queue = deque([(tuple(), self._store)])
        while queue:
            queue.append(None)  # invariant: level end
            yield self._items_partial(queue)

    def items(self):  # iterate breadth-first
        return chain.from_iterable(self.items_nested())

    _defaultkey = None


class NestedDict:
    def __init__(self):
        fn = lambda: defaultdict(fn)
        self._store = fn()

    def __getitem__(self, itemseq):
        current = self._store
        for item in itemseq:
            current = current[item]
        ret = current.get(self._defaultkey)
        if ret is not None:
            return ret
        raise KeyError

    def __setitem__(self, itemseq, value):
        current = self._store
        for item in itemseq:
            current = current[item]
        current[self._defaultkey] = value

    def items(self):  # iterate breadth-first
        stack = deque([(tuple(), self._store)])
        while stack:
            prefix, store = stack.popleft()
            for node, val in store.items():
                if node is self._defaultkey:
                    yield prefix, val
                elif val:
                    stack.append((prefix + (node,), val))

    def keys(self):  # iterate breadth-first
        for path, val in self.items():
            yield path

    def values(self):  # iterate breadth-first
        stack = deque([self._store])
        while stack:
            store = stack.popleft()
            for node, val in store.items():
                if node is self._defaultkey:
                    yield val
                elif val:
                    stack.append(val)

    _defaultkey = None


class DefaultList(list):
    """A list class with default values designed for rare index misses (otherwise, don't use exceptions)"""
    def __init__(self, fx):
        self._fx = fx

    def __setitem__(self, index, value):
        try:
            list.__setitem__(self, index, value)
        except IndexError:
            while len(self) < index:
                self.append(self._fx())
            self.append(value)

    def __getitem__(self, index):
        try:
            list.__getitem__(self, index)
        except IndexError:
            while len(self) <= index:
                self.append(self._fx())
            return list.__getitem__(self, index)

def handle_broken_pipe():
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


if __name__ == "__main__":
    pass
