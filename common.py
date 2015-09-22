#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains helper functions and types.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import numpy as np
import math
from numpy.testing import assert_approx_equal
from scipy.special import binom
from operator import itemgetter
from itertools import count, filterfalse, chain
from collections import defaultdict, deque
from sys import stderr, stdout


# common data types
prob_type = np.float16


class UniversalData(list):  # TODO: rename GenericData
    def __init__(self, *args, sizes: "1d NumPy array", **kwargs):
        super(UniversalData, self).__init__(*args, **kwargs)
        try:
            self.sizes = np.asarray(sizes, dtype=self.size_type)
        except TypeError:
            self.sizes = np.fromiter(sizes, dtype=self.size_type)

    def deposit(self, features):
        # self.names.append(name)
        for d, f in zip(self, features):
            # print(d, f)
            d.deposit(f)

    def prepare(self):
        for d in self:
            d.prepare()
        return self
        #return [d.prepare() for d in self]  # TODO: return self without conversion to list by map

    # @property
    # def sizes(self):  # transitional
    #     return self[0].sizes
    #     return self[0].sizes

    @property
    def num_data(self):
        if not super(UniversalData, self).__len__():
            return 0

        # print(self, file=stderr)
        num = self[0].num_data
        assert num == len(self[0])
        for l in self[1:]:
            assert(l.num_data == num)
        return num

    @property
    def num_features(self):
        return super(UniversalData, self).__len__()

    def __len__(self):
        return self.num_data   # TODO: select an intuitive convention for this

    size_type = np.uint32


class UniversalModel(list):  # TODO: rename GenericModel, implement update() and maximize_likelihood()
    def __init__(self, weights, *args, **kw):
        super(UniversalModel, self).__init__(*args, **kw)
        # self._sharpness = sharpness  # TODO: move sharpness outside supermodel to EM and normalize this?
        self.weights = np.asarray(weights)[:, np.newaxis, np.newaxis]
        # self.weights = np.repeat(self._sharpness/float(len(self)), len(self))
        # self.weights = flat_priors(len(self))[:, np.newaxis, np.newaxis]  # TODO: decide explicit likelihood type?

    @property
    def names(self):
        if len(self) > 1:
            component_names = list(zip(*[m.names for m in self]))
            return [",".join(t) for t in component_names]
        return self[0].names

    @property
    def num_components(self):  # transitional
        if not len(self):
            return 0
        cluster_num = self[0].num_components
        assert np.all(np.equal(cluster_num, [model.num_components for model in self[1:]]))
        return cluster_num

    def log_likelihood(self, data):
        assert self.weights.size == len(self)

        ll_per_model = np.asarray([m.log_likelihood(d) for (m, d) in zip(self, data)])
        #print(ll_per_model.shape, file=stderr)
        s = np.mean(np.exp(ll_per_model), axis=1)
        l = np.sum(ll_per_model, axis=1)
        #print(s.shape, file=stderr)
        for m, mvec, lvec in zip(self, s, l):
            print(m._short_name, "***", pretty_probvector(mvec), "***", pretty_probvector(lvec), file=stderr)
        # total_ll_per_model = np.asarray([total_likelihood(ll) for ll in ll_per_model])
        # ll_joint = (self._sharpness * self.weights * ll_per_model).sum(axis=0, keepdims=False)
        # total_ll_joint = total_likelihood(ll_joint)

        # determine likelihood by joint posterior of data given clusters
        # normalized_ll_per_model = np.asarray([np.log(exp_normalize(ll)) for ll in self.weights * ll_per_model])

        # Determine weights
        # for ll in ll_per_model:
        #     print_probvector(ll[0], file=stderr)
        # print_probvector(total_ll_per_model, file=stderr)
        # stderr.write("%i\n" % total_ll_joint)
        # self.weights = exp_normalize_1d(total_ll_per_model)[:, np.newaxis, np.newaxis]
        # stderr.write("LOG ECM #: -- | LL: %i | weights: %s\n" % (total_ll_joint, pretty_probvector(self.weights)))

        return (self.weights * ll_per_model).sum(axis=0, keepdims=False)
        # return ll_joint

        # start with equally weighted likelihoods
        # loglike = self.weights[0] * self[0].log_likelihood(data[0])
        # for m, d, w in zip(self[1:], data[1:], self.weights[1:]):
        #     m_loglike = m.log_likelihood(d)
        #     loglike += w * m_loglike
        # return loglike

    def maximize_likelihood(self, responsibilities, data, cmask=None):
        tmp = (m.maximize_likelihood(responsibilities, d, cmask) for m, d in zip(self, data))  # TODO: needs to know weights?
        # ll_per_model = np.asarray([m.log_likelihood(d) for (m, d) in zip(self, data)])

        # TODO: assert that total likelihood calculation is consistent

        # # ECM variant for weight optimization
        # if len(self.weights > 1):
        #     iteration = count()
        #     total_joint = total_likelihood(self._sharpness * np.asarray([np.log(exp_normalize(ll)) for ll in self.weights * ll_per_model]).sum(axis=0, keepdims=False))
        #     while next(iteration) < 10:
        #         weights_new = random_probarray(len(self))[:, np.newaxis, np.newaxis]  # TODO: suggest from neighborhood instead of random
        #         # print_probvector(weights_new, stderr)
        #         total_joint_new = total_likelihood(self._sharpness * np.asarray([np.log(exp_normalize(ll)) for ll in self.weights * ll_per_model]).sum(axis=0, keepdims=False))
        #         # stderr.write("LOG ECM #: -- | LL: %i | Δ: %.2f | weights: %s\n" % (total_joint_new, total_joint_new - total_joint, pretty_probvector(weights_new)))
        #         if total_joint_new > total_joint:
        #             self.weights = weights_new
        #             stderr.write("LOG ECM #: -- | LL: %i | Δ: %.2f | weights: %s\n" % (total_joint_new, total_joint_new - total_joint, pretty_probvector(self.weights)))
        #             break
        #     # print_probvector(self.weights, stderr)

        # maximize the weights as well, need to return likelihood in previous function call?
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

load_data = lambda lines, store: store.parse(parse_lines(lines))
load_data_file = load_data  # transitional alias


def assert_probmatrix(mat):
    is_sum = mat.sum()
    should_sum = mat.shape[0]
    assert_approx_equal(is_sum, should_sum, significant=0)
    [assert_approx_equal(rowsum, 1., significant=1) for rowsum in mat.sum(axis=1)]
    # assert(np.all(1. - mat.sum(axis=1) <= 0.0001))
    # print np.all(mat.sum(axis=1) == 1.)


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


def responsibilities_from_seeds(seed_indices, num_data):
    responsibilities = np.zeros((num_data, len(seed_indices)), dtype=prob_type)
    for i, s in enumerate(seed_indices):
        responsibilities[list(s), i] = 1.  # TODO: index with numpy array instead of list?
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
    for line in iterable:
        if line and line[0] == "#":  # TODO: factorize
            continue
        yield line.rstrip().split(" ")

load_data_sizes = lambda lines: (int(line.rstrip()) for line in lines)

load_seqnames = lambda lines: (line.rstrip() for line in lines)



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
    return np.asarray([np.math.factorial(i) for i in vec])


# debug function
def log_array(vec):
    return np.asarray([math.log(i) for i in vec], dtype=float)


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
        fn = lambda: defaultdict(fn)
        self._store = fn()
        self._size = 0

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


if __name__ == "__main__":
    pass
