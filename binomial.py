#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 This file holds all the functions and types necessary for probabilistic modelling of differential (read) coverage.
 We use a Binomial density to model the coverage per position which also handles low count values.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import common
import numpy as np
from sys import argv, exit, stdin, stdout, stderr, exit

# count data type
frequency_type = np.int32
logtype = np.float64  # TODO: adjust according to maximum values
logfile = open("coverage.log", "w")

# TODO: clear dependency on scipy
from scipy.special import gammaln
logfactorial = lambda n, k: gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)


class Data:
    def __init__(self, sizes):
        self._covmeans = []  # TODO: use deque() for large append-only lists
        self.covmeans = None
        self.covmeanstotal = None
        self.conterm = None
        self.sizes = sizes

    def deposit(self, features):  # TODO: improve data parsing and handling
        coverage = np.array(features, dtype=self.mean_coverage_type)
        self._covmeans.append(coverage)

    def parse(self, inseq):  # TODO: add load_data from generic with data-specific parse_line function
        for entry in inseq:  # space-separated list of sample mean coverage
            self.deposit(entry.split(" "))
        return self.prepare()

    def prepare(self):
        self.covmeans = np.vstack(self._covmeans)
        self.covmeanstotal = self.covmeans.sum(axis=1, keepdims=True)
        self.conterm = np.asarray(logfactorial(self.covmeanstotal, self.covmeans).sum(axis=1, keepdims=True), dtype=common.logprob_type)

        assert(np.all(self.covmeanstotal > 0))  # TODO: what about zero observation in all samples
        return self

    @property
    def num_features(self):
        return self.covmeans.shape[1]

    @property
    def num_data(self):
        return self.covmeans.shape[0]

    def __len__(self):
        return self.num_data  # TODO: select an intuitive convention for this

    mean_coverage_type = np.float32


class Model:
    def __init__(self, params, initialize=True):
        #print(params.shape, file=stderr)
        self.params = np.array(params).T  # TODO: why pass transposed?
        self.standard_deviation = None
        self._params_sum = None
        self._params_log = None
        self._params_complement_log = None

        if initialize:
            self.update()

    def update(self):
        self._params_log = np.asarray(np.log(self.params), dtype=common.logprob_type)
        self._params_complement_log = np.asarray(np.log(1. - self.params), dtype=common.logprob_type)
        return False  # indicates whether a dimension change occurred

    def log_likelihood(self, data, normalize=True):  # TODO: check and adjust formula
        assert data.num_features == self.num_features
        term1 = np.dot(data.covmeans, self._params_log)  # TODO: scipy special.xlogy(k, p)?
        assert np.all(~np.isnan(term1))
        term2 = np.dot(data.covmeanstotal - data.covmeans, self._params_complement_log)  # TODO: scipy  special.xlog1py(n-k, -p)?
        assert np.all(~np.isnan(term2))
        loglike = np.asarray(term1 + term2 + data.conterm, dtype=common.logprob_type)
        #loglike = loglike/self.num_features  # normalize by number of samples  # deprecated due to normalization

        if normalize:
            loglike = loglike/self.standard_deviation  # normalize by setting stdev to one
            stderr.write("Normalizing class likelihoods by factors %s\n" % common.pretty_probvector(1/self.standard_deviation))

        common.write_probmatrix(loglike, file=logfile)

        assert np.all(loglike <= .0)
        return loglike

    def get_labels(self, indices=None):
        if not indices:
            indices = list(range(self.params.shape[1]))
        for i in indices:
            yield "-".join(("%i" % round(v) for v in np.asarray(self.params)[:, i]))

    def maximize_likelihood(self, responsibilities, data, cmask=None):  # TODO: adjust
        # TODO: input as combined weights, not responsibilities and data.sizes
        size_weights = np.asarray(data.sizes/data.sizes.sum(), dtype=common.prob_type)
        #size_weights = data.sizes

        if cmask is None or cmask.shape == () or np.all(cmask):
            weights = responsibilities * size_weights
        else:
            weights = responsibilities[:, cmask] * size_weights

        stderr.write("weights dtype: %s\n" % weights.dtype)

        weighted_meancoverage_samples = np.dot(data.covmeans.T, weights)  # TODO: use np.average?
        weighted_meancoverage_total = np.dot(data.covmeanstotal.T, weights)  # TODO: use np.average? simplify?

        stderr.write("meancov dtype: %s/%s\n" % (weighted_meancoverage_samples.dtype, weighted_meancoverage_total.dtype))

        pseudocount = 0.0000000001  # TODO: refine
        self.params = np.asarray((weighted_meancoverage_samples + pseudocount) / (weighted_meancoverage_total + pseudocount),
                                 dtype=common.prob_type)  # introduced pseudocounts
        stderr.write("params dtype: %s\n" % self.params.dtype)

        dimchange = self.update()  # create cache for likelihood calculations
        ll = self.log_likelihood(data, normalize=False)
        stderr.write("ll dtype: %s\n" % ll.dtype)
        self.standard_deviation = np.asarray(np.sqrt(common.weighted_variance(ll, weights)), dtype=common.logprob_type)
        stderr.write("Weighted stdev was: %s\n" % common.pretty_probvector(self.standard_deviation))
        return dimchange

    @property
    def num_components(self):
        return self.params.shape[1]

    @property
    def num_features(self):
        return self.params.shape[0]

    @property
    def names(self):
        return list(self.get_labels())

    _short_name = "BI_model"


def empty_model(cluster_number, initial_data, **kwargs):
    assert cluster_number > 0
    assert type(initial_data) == Data
    params = np.zeros(shape=(cluster_number, initial_data.num_features), dtype=common.prob_type)
    return Model(params, **kwargs)


def random_model(cluster_number, initial_data, **kwargs):
    assert cluster_number > 0
    assert type(initial_data) == Data
    params = np.random.rand(cluster_number, initial_data.num_features)
    params /= params.sum(axis=1, keepdims=True)
    return Model(params, **kwargs)


def load_data_file(filename, **kwargs):
    d = Data(**kwargs)
    return common.load_data_file(filename, d)
