#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# This file holds all the functions and types necessary for probabilistic modelling of (read) coverage.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from common import *
import numpy as np
from sys import argv, exit, stdin, stdout, stderr, exit

# count data type
frequency_type = np.int32
logfile = open("coverage.log", "w")


class Data:
    def __init__(self, samples):
        self.samples = []
        self._samplename2index = {}
        self._covsums = []
        self._seqlens = []
        self._sum_log_fac_covs = []  # TODO: remove optional part
        self._intialize_samples(samples)
        self._zero_count_vector = np.zeros(len(samples), dtype=frequency_type)
        self._zero_count_vector_uint = np.zeros(len(samples), dtype=frequency_type)
        self.covsums = None
        self.sizes = None
        self.covmeans = None
        self.facterm = None

    def _intialize_samples(self, samples):
        for i, sample in enumerate(samples):
                self._samplename2index[sample] = i
                self.samples.append(sample)

    def deposit(self, features):  # TODO: improve data parsing and handling
        if not features:
            print("empty coverage features set deposit!", file=stderr)

        length = 0  # TODO: transfer length to UniversalData object and use only average coverage
        row_covsums = self._zero_count_vector.copy()
        row_facsums = self._zero_count_vector_uint.copy()  # TODO. remove optional term

        for sample_name, sample_coverage in features:
            coverage = list(map(self.coverage_type, sample_coverage))  # TODO: use sparse numpy objects...
            # print(coverage)
            # feature_list.append((sample_name, coverage))
            try:
                index = self._samplename2index[sample_name]
                row_covsums[index] = np.sum(coverage)
                length = len(coverage)
                # print(max(coverage), file=stderr)
                tmp = log_array(factorial_array(coverage))/length
                # print(tmp.min(), file=stderr)
                # print(tmp.max(), file=stderr)
                row_facsums[index] = tmp.sum()  # TODO: replace by optimize code, if not removed!!!
                assert(row_covsums[index])
            except KeyError:
                pass
                # stderr.write("Feature with sample name \"%s\" ignored.\n" % sample)
        # if length:  # only process non-empty data
        self._covsums.append(row_covsums)
        self._seqlens.append(length)
        self._sum_log_fac_covs.append(row_facsums)  # TODO: simplify or remove (one term per datum)
        # self.names.append(name)

    def parse(self, inseq):  # TODO: add load_data from generic with data-specific parse_line function
        for entry in inseq:
            feature_list = []
            for sample_group in entry.split(" "):
                sample_name, sample_coverage = sample_group.split(":", 2)[:2]
                feature_list.append((sample_name, sample_coverage.split(",")))
            self.deposit(feature_list)
        return self.prepare()

    def prepare(self):
        self.covsums = np.vstack(self._covsums)
        self.sizes = np.array(self._seqlens, dtype=frequency_type)[:, np.newaxis]
        self.covmeans = self.covsums / self.sizes
        self.facterm = sum(self._sum_log_fac_covs)  # TODO: do not really need to calculate and store previous intermediate values
        # assert(np.all(self.covsums.sum(axis=1) > 0))  # zero observation in all samples might be possible TODO: check cases
        return self

    @property
    def num_features(self):
        return self.covsums.shape[1]

    def __len__(self):
        return self.covsums.shape[0]

    coverage_type = np.uint32


class Model:
    def __init__(self, params, initialize=True, pseudocount=False):  # TODO: does pseudocount make sense here?
        if pseudocount:
            self.params = np.array(params + 1).T
            self._pseudocount = True
        else:
            self.params = np.array(params).T
            self._pseudocount = False

        self._params_sum = None
        self._params_log = None

        if initialize:
            self.update()

    def update(self):
        # if not np.all(self.params):
            # print >>stderr, "some cluster in some sample wasn't observed:", self.params
        self._params_sum = self.params.sum(axis=0, keepdims=True)
        self._params_log = np.log(self.params)
        return False  # indicates whether a dimension change occurred

    def log_likelihood(self, data):  # TODO: check and adjust formula
        # term1 = np.dot(data.covsums, self._params_log)  # sum of data coverage version
        term1 = np.dot(data.covmeans, self._params_log)  # mean coverage version
        # print("term1 shape is %ix%i" % term1.shape)
        # term2 = np.dot(data.sizes, self._params_sum)  # sum of data coverage version
        term2 = self._params_sum  # mean coverage version
        # print("term2 shape is %ix%i" % term2.shape)
        loglike = term1 - term2 - data.facterm  # last term is optional!
        # print >>stderr, loglike
        return loglike

    def get_labels(self, indices=None):
        if not indices:
            indices = list(range(self.params.shape[1]))
        for i in indices:
            yield "-".join(("%i" % round(v) for v in np.asarray(self.params)[:, i]))

    def maximize_likelihood(self, responsibilities, data, cmask=None):  # TODO: adjust
        if cmask is None or cmask.shape == () or np.all(cmask):
            weighted_coverage_sum = np.dot(data.covsums.T, responsibilities)
            weighted_length_sum = np.dot(data.sizes.T, responsibilities)
        else:
            # print >>stderr, "shape of responsibilities vector:", responsibilities.shape
            weighted_coverage_sum = np.dot(data.covsums.T, responsibilities[:, cmask])
            weighted_length_sum = np.dot(data.sizes.T, responsibilities[:, cmask])

        self.params = weighted_coverage_sum / weighted_length_sum
        return self.update()

    @property
    def components(self):
        return self.params.shape[1]

    @property
    def names(self):
        return list(self.get_labels())

    # @property
    # def features_used(self):
        # return sum(self._fmask)

    _short_name = "NB_model"


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


def random_model(component_number, feature_number, low, high):  # TODO: make generic
    params = np.random.randint(low, high, (component_number, feature_number))
    return Model(params)


def empty_model(component_number, feature_number):  # TODO: make generic
    params = np.zeros(shape=(component_number, feature_number), dtype=frequency_type)
    return Model(params)