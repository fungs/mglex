#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 This file holds all the functions and types necessary for probabilistic modelling of differential (read) coverage.
 We use a Binomial density to model the coverage per position which also handles low count values.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from common import *
import numpy as np
from sys import argv, exit, stdin, stdout, stderr, exit

# count data type
frequency_type = np.int32
logtype = np.float64  # TODO: adjust according to maximum values
logfile = open("coverage.log", "w")


class Data:
    def __init__(self, samples):  # TODO: use deque() for large append-only lists
        assert len(samples) > 1
        self.samples = []
        self._samplename2index = {}
        self._covsums = []
        self._seqlens = []
        self._intialize_samples(samples)
        self._zero_count_vector = np.zeros(len(samples), dtype=frequency_type)
        self._zero_count_vector_uint = np.zeros(len(samples), dtype=frequency_type)
        self._conterm = np.zeros(len(samples), dtype=logtype)
        self._samples_ignored = set()
        self.covsums = None
        self.sizes = None
        self.covmeans = None
        self.conterm = None
        self.covmeanstotal = None

    def _intialize_samples(self, samples):
        for i, sample in enumerate(samples):
                self._samplename2index[sample] = i
                self.samples.append(sample)

    def deposit(self, features):  # TODO: improve data parsing and handling
        if not features:
            print("empty coverage features set deposit!", file=stderr)

        length = 0  # TODO: transfer length to UniversalData object and use only average coverage
        row_covsums = self._zero_count_vector.copy()
        # row_facsums = self._zero_count_vector_uint.copy()  # TODO. remove optional term
        covmat = None
        indexorder = []

        for sample_name, sample_coverage in features:
            if sample_name in self._samples_ignored:
                continue
            try:
                index = self._samplename2index[sample_name]
            except KeyError:
                stderr.write("Features for sample \"%s\" ignored.\n" % sample_name)
                self._samples_ignored.add(sample_name)
                continue

            # constant term part can be removed when not used, it takes initialization time but no time for other operations
            coverage = np.array(sample_coverage, dtype=self.coverage_type)  # TODO: use sparse numpy objects...
            row_covsums[index] = np.sum(coverage)
            length = coverage.size
            assert(row_covsums[index])  # TODO: what does this mean? what about zero coverage in repliate?
            indexorder.append(index)

            try:  # TODO: bad style, do typesafe operations instead; how to best grow the row matrix
                covmat = np.vstack((covmat, coverage))
            except ValueError:
                covmat = np.array([coverage])

        assert indexorder  # empty data not allowed, must have some coverage in some sample

        covsums_positional = covmat.sum(axis=0)  # TODO: add up instead of array creation; finally remove all

        for i, row in zip(indexorder, covmat):
            self._conterm[i] += np.sum(log_array(binom_array(covsums_positional, row)))

        self._covsums.append(row_covsums)
        self._seqlens.append(length)


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
        self.covmeanstotal = self.covsums.sum(axis=1, keepdims=True) / self.sizes
        self.conterm = self._conterm  # TODO: do not really need to calculate
        assert(np.all(self.covmeanstotal > 0))  # zero observation in all samples might be possible TODO: check cases, optimize code
        return self

    @property
    def num_features(self):
        return self.covsums.shape[1]

    @property
    def num_data(self):
        return self.covsums.shape[0]

    def __len__(self):
        return self.num_data  # TODO: select an intuitive convention for this

    coverage_type = np.uint32


class Model:
    def __init__(self, params, initialize=True):
        #print(params.shape, file=stderr)
        self.params = np.array(params).T  # TODO: why pass transposed?
        self._params_sum = None
        self._params_log = None
        self._params_complement_log = None

        if initialize:
            self.update()

    def update(self):
        self._params_log = np.log(self.params)
        self._params_complement_log = np.log(1. - self.params)
        return False  # indicates whether a dimension change occurred

    def log_likelihood(self, data):  # TODO: check and adjust formula
        term1 = np.dot(data.covmeans, self._params_log)  # mean coverage version
        # print("term1 shape is %ix%i" % term1.shape)
        # term2 = np.dot(data.sizes, self._params_sum)  # sum of data coverage version
        term2 = np.dot(data.covmeanstotal - data.covmeans, self._params_complement_log)  # mean coverage version
        # print("term2 shape is %ix%i" % term2.shape)
        loglike = term1 + term2 #+ data.conterm  # constant term is not necessary for EM TODO: fix conterm (need values per sample), should not be needed at all
        #loglike = loglike * (self.num_components-1)/self.num_components  # TODO: renormalization for  dependent dimension is a constant, should not be needed
        # print >>stderr, loglike
        return loglike

    def get_labels(self, indices=None):
        if not indices:
            indices = list(range(self.params.shape[1]))
        for i in indices:
            yield "-".join(("%i" % round(v) for v in np.asarray(self.params)[:, i]))

    def maximize_likelihood(self, responsibilities, data, cmask=None):  # TODO: adjust
        if cmask is None or cmask.shape == () or np.all(cmask):
            weights = responsibilities*data.sizes
        else:
            # print >>stderr, "shape of responsibilities vector:", responsibilities.shape
            weights = responsibilities[:, cmask] * data.sizes

        weighted_meancoverage_samples = np.dot(data.covmeans.T, weights)
        weighted_meancoverage_total = np.dot(data.covmeanstotal.T, weights)

        self.params = weighted_meancoverage_samples / weighted_meancoverage_total
#        print_probmatrix(self.params, stderr)
        return self.update()

    @property
    def num_components(self):
        return self.params.shape[1]

    @property
    def names(self):
        return list(self.get_labels())

    # @property
    # def features_used(self):
        # return sum(self._fmask)

    _short_name = "BI_model"


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


def empty_model(component_number, feature_number):  # TODO: make generic
    params = np.zeros(shape=(component_number, feature_number), dtype=prob_type)
    return Model(params)


def random_model(component_number, feature_number):  # TODO: make generic
    params = np.random.rand(component_number, feature_number)
    params /= params.sum(axis=1, keepdims=True)
    return Model(params)   

