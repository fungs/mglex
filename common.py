#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains helper functions and types.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import numpy as np
from collections import Iterable
from numpy.testing import assert_approx_equal
from operator import itemgetter
from itertools import count, filterfalse
import unittest
from sys import stderr, stdout


# common data types
prob_type = np.float16


class UniversalData(list):  # TODO: rename GenericData
    def __init__(self, *args, **kw):
        super(UniversalData, self).__init__(*args, **kw)
        # self.names = []

    def deposit(self, features):
        # self.names.append(name)
        for d, f in zip(self, features):
            print(d, f)
            d.deposit(f)

    def prepare(self):
        return [d.prepare() for d in self]  # TODO: return self without conversion to list by map

    @property
    def sizes(self):  # transitional
        return self[0].sizes

    @property
    def num_data(self):
        if len(self):
            # print >>stderr, map(len, self)
            num = len(self[0])
            for l in self[1:]:
                assert(len(l) == num)
            return num
        return 0


class UniversalModel(list):  # TODO: rename GenericModel, implement update() and maximize_likelihood()
    def __init__(self, *args, **kw):
        super(UniversalModel, self).__init__(*args, **kw)

    @property
    def names(self):
        if len(self) > 1:
            component_names = list(zip(*[m.names for m in self]))
            return [",".join(uniq(t)) for t in component_names]
        return self[0].names

    @property
    def components(self):  # transitional
        return self[0].components

    def log_likelihood(self, data):
        loglike = self[0].log_likelihood(data[0])
        for m, d in zip(self[1:], data[1:]):
            m_loglike = m.log_likelihood(d)
            loglike += m_loglike
        return loglike

    def maximize_likelihood(self, responsibilities, data, cmask=None):
        tmp = [m.maximize_likelihood(responsibilities, d, cmask) for m, d in zip(self, data)]
        return any(tmp)


def parse_lines(lines):
    for line in lines:
        if not line or line[0] == "#":  # skip empty lines and comments
            continue
        yield line.rstrip()


# def parse_lines_comma(lines):
#     for fields in parse_lines(lines):
#         fields[1:] = [s.split(",") for s in fields[1:]]  # TODO: remove name column from input
#         yield tuple(fields)


# def load_data_tuples(inseq, store):  # TODO: make dependent on data class
#         names = []
#         for record in inseq:
#             names.append(record[0])
#             features = record[1:]
#             if len(features) > 0 and isinstance(features[0], Iterable) and not isinstance(store, UniversalData):  # hack to make this work for UniversalData and Data
#                 store.parse_data(*features)
#             else:
#                 store.deposit(features)
#         return names, store.prepare()


#load_data = lambda lines, store: load_data_tuples(parse_lines_comma(lines), store)

def load_data_file(fh, store):
    store.parse(parse_lines(fh))


def assert_probmatrix(mat):
    is_sum = mat.sum()
    should_sum = mat.shape[0]
    assert_approx_equal(is_sum, should_sum, significant=0)
    [assert_approx_equal(rowsum, 1., significant=0) for rowsum in mat.sum(axis=1)]
    # assert(np.all(1. - mat.sum(axis=1) <= 0.0001))
    # print np.all(mat.sum(axis=1) == 1.)


def approx_equal(v1, v2, precision):
    if type(v1) == type(v2) == np.ndarray:
        if v1.shape != v2.shape:
            return False
        return (abs(v1-v2) < precision).all()
    return abs(v1-v2) < precision


assert_probarray = lambda v: assert_approx_equal(v.sum(), 1.)


def argmax(s, n=1):
    get_second = itemgetter(1)
    max_store = sorted(list(enumerate(s[:n])), key=get_second, reverse=True)
    for e in zip(count(n), s[n:]):
        max_store = sorted(max_store + [e], key=get_second, reverse=True)[:n]
    if n == 1:
        return max_store[0]
    return max_store


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


def log_fac(i):
    r = .0
    while i > 0:
        r += np.exp(i)
        i -= 1
    return r


def seeds2indices(seqnames, seeds):
    # a) build a dictionary for the seeds for fast lookup
    name2cluster = {}
    for i, names in enumerate(seeds):
        for n in names:
            name2cluster[n] = i

    seed_indices = [[] for i in range(len(seeds))]

    # b) determine indices of seeds
    for i, name in enumerate(seqnames):
        cluster_index = name2cluster.get(name, None)
        if cluster_index is not None:
            seed_indices[cluster_index].append(i)
    return seed_indices


def responsibilities_from_seeds(seed_indices, num_data):
    responsibilities = np.zeros((num_data, len(seed_indices)), dtype=prob_type)
    for i, s in enumerate(seed_indices):
        responsibilities[s, i] = 1.
    return responsibilities


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
    seeds = []
    for line in iterable:
        if line and line[0] == "#":
            continue
        seeds.append(line.rstrip().split(" "))
    return seeds


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

from itertools import filterfalse


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


def print_probmatrix(mat, out=stdout):
    for row in np.asarray(mat):
        out.write("\t".join(["%.2f" % i for i in row]))
        out.write("\n")

print_predictions = lambda mat: print_probmatrix(np.absolute(np.log(mat)))


if __name__ == "__main__":
    pass
