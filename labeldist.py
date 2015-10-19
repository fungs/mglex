#!/usr/bin/env python3

u"""
 The types and methods for describing the distribution of label-type data. We use a modified Naive Bayesian Model
 which  considers hierarchical labels using a weighting scheme. However, the weighting of the individual labels
 is handled externally which leaves much freedom for shaping the actual PMF. Weights could for instance be set by
 consideration phylogenetic distances.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from common import *
import numpy as np
from collections import deque
from sys import argv, exit, stdin, stdout, stderr, exit

# label data type
logtype = np.float64  # TODO: adjust according to maximum values
logfile = open("labeldist.log", "w")
label_index_type = np.uint16  # TODO: check range

# TODO: return abstract shape object from Data with parameters to construct Model

class Data:  # TODO: use deque() for large append-only lists

    support_type = np.uint32  # TODO: check range

    def __init__(self):
        self._label_mapping = NestedCountIndex()
        self._labels = []  # TODO: operate on separate deque objects?
        self.num_features = 0
        self.labels = None
        self.levelindex = []

    def deposit(self, features):
        features = tuple((self._label_mapping[path], self.support_type(support)) for path, support in features)
        self._labels.append(features)

    def parse(self, inseq):  # TODO: add load_data from generic with data-specific parse_line function
        for line in inseq:
            feature_list = []
            if line:
                for entry in line.split(","):
                    # print(entry)
                    path, support = entry.split(":", 2)[:2]
                    path = path.split(".")
                    feature_list.append((path, support))
            self.deposit(feature_list)
        return self.prepare()

    def prepare(self):
        # calculate internal (ordered) label index
        index2index = np.empty(len(self._label_mapping), dtype=label_index_type)
        newindex = 0
        for levelindex, level in enumerate(self._label_mapping.values_nested()):  # skip root level?
            for oldindex in level:
                index2index[oldindex] = newindex
                newindex += 1
            self.levelindex.append(newindex)  # whenever a deeper level is reached, the index is saved
        self.num_features = len(self._label_mapping)

        # reset temporary index which is not needed any more
        self._label_mapping = NestedCountIndex()

        # replace old by new indices and create numpy arrays inplace TODO: make this two real 2d arrays
        for i, features in enumerate(self._labels):
            index_col = np.empty(len(features), dtype=label_index_type)
            support_col = np.empty(len(features), dtype=self.support_type)
            for j, (index_orig, support) in enumerate(features):
                index_col[j] = index2index[index_orig]
                support_col[j] = support
            self._labels[i] = (index_col, support_col)

        del index2index  # runs out of scope and should be garbage-collected anyway

        self.labels = self._labels
        self._labels = []

        # print(self.levelindex)
        return self

    @property
    def num_data(self):
        return len(self.labels)

    def __len__(self):
        return self.num_data


class Model:

    support_type = np.float32  # TODO: check range

    def __init__(self, params, levelindex, initialize=True, pseudocount=True):
        self.params = np.array(params, dtype=self.support_type)  # TODO: use large unsigned integer first, then cut down
        self._levelindex = np.asarray(levelindex, dtype=label_index_type)
        self.levelsum = np.empty(params.shape, dtype=self.support_type)
        self._pseudocount = pseudocount

        if initialize:
            self.update()

    def update(self):
        # print("params")
        # print(self.params.shape)
        # print(self.params)
        for i, j in zip(self._levelindex, self._levelindex[1:]):  # TODO: advanced slicing with np.r_?
            self.levelsum[i:j] = self.params[i:j].sum(axis=0)
        # print("levelsum")
        # print(self.levelsum.shape)
        # print(self.levelsum)
        return False  # indicates whether a dimension change occurred

    def log_likelihood(self, data):  # TODO: check
        loglike = np.empty((len(data), self.num_components), dtype=logtype)
        for i, (indexcol, supportcol) in enumerate(data.labels):  # TODO: vectorize 3d?

            if not indexcol.size:  # no label == no observation == perfect fit
                loglike[i] = 0
                continue

            denominator = np.dot(supportcol, self.levelsum[indexcol])
            assert np.all(denominator != 0.0)
            numerator = np.dot(supportcol, self.params[indexcol])


            # if not np.all(numerator != 0.):
            #     print(pretty_probvector(numerator), file=stderr)
            #     print(pretty_probvector(denominator), file=stderr)
            #     print(pretty_probvector(indexcol), file=stderr)
            #     print(pretty_probvector(supportcol), file=stderr)

            # if not denominator.all():  # TODO: turn back into assertion
            #     print("datum: %i\n  numerator: %s /\n denominator %s" % (i, numerator, denominator), file=stderr)
            #     print(indexcol)
            #     print(supportcol)
            #     print(self.levelsum[indexcol])
            #     print(denominator)
            #     exit(1)

            #print(pretty_probvector(numerator),file=stderr)
            #print(pretty_probvector(denominator), file=stderr)

            if self._pseudocount:
                probs = (numerator+1)/(denominator+1)
            else:
                probs = numerator/denominator

            ll = np.log(probs)  # TODO: or log - log
            assert (ll <= 0.).all()
            loglike[i] = ll

        assert np.all(loglike <= .0)
        return loglike

    def get_labels(self, indices=None):
        if not indices:
            for i in self.params.argmax(axis=0):
                yield str(i)
        else:
            for i in self.params[indices].argmax(axis=0):
                yield str(i)

    def maximize_likelihood(self, responsibilities, data, cmask=None):  # TODO: adjust
        if not (cmask is None or cmask.shape == () or np.all(cmask)):  # cluster reduction
            # print(cmask)
            # print(self.params.shape)
            self.params = self.params[:, cmask]
            responsibilities = responsibilities[:, cmask]
            self.params = self.params[:, cmask]

        self.params[:] = 0  # zero out values

        for res, (index_col, support_col) in zip(responsibilities, data.labels):
            # print_probvector(res)
            # print(index_col)
            # print(self.params.shape)
            # print(support_col[:, np.newaxis].shape)
            # print(res.T)
            # print(np.vdot(support_col, res).shape)
            self.params[index_col] += np.dot(support_col[:, np.newaxis], res[np.newaxis, :])  # TODO: check shape match
        return self.update()

    @property
    def num_components(self):
        return self.params.shape[1]

    @property
    def names(self):
        return list(self.get_labels())

    _short_name = "LD_model"


def load_model(instream):
    all_clists = []
    # samples = input.next().rstrip().split("\t")
    for line in instream:
        if not line or line[0] == "#":
            continue
        clist = line.rstrip().split("\t")
        if clist:
            all_clists.append(list(map(int, clist)))
    return Model(all_clists)


def load_data(input, samples):  # TODO: add load_data from generic with data-specific parse_line function
    store = Data(samples)
    for line in input:
        if not line or line[0] == "#":  # skip empty lines and comments
            continue
        seqname, coverage_field = line.rstrip().split("\t", 2)[:2]
        feature_list = []
        for sample_group in coverage_field.split(" "):
            sample_name, coverage = sample_group.split(":", 2)[:2]
            coverage = list(map(int, coverage.split(",")))  # TODO: use sparse numpy objects...
            feature_list.append((sample_name, coverage))
        store.deposit(seqname, feature_list)
    return store.prepare()


def empty_model(component_number, feature_number, levelindex):  # TODO: make generic
    params = np.zeros(shape=(feature_number, component_number), dtype=prob_type)
    return Model(params, levelindex)


def random_model(component_number, feature_number, levelindex, low=0, high=None):  # TODO: make generic
    params = np.random.random_integers(low=low, high=high, size=(feature_number, component_number))
    return Model(params, levelindex)
