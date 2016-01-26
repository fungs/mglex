#!/usr/bin/env python3

u"""
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
factorial = lambda n, k: gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)


class Context(object):
    """Container for information which is shared between Data and Model"""

    def __init__(self):
        self.num_features = None


class Data(object):
    def __init__(self, sizes, context=Context()):
        self.context = context
        self._covmeans = []  # TODO: use deque() for large append-only lists
        self.covmeans = None
        self.covmeanstotal = None
        self.conterm = None
        self.sizes = sizes  # TODO: change to general weights argument in maximize_likelihood()

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
        self.conterm = factorial(self.covmeanstotal, self.covmeans).sum(axis=1, keepdims=True)

        assert(np.all(self.covmeanstotal > 0))  # TODO: what about zero observation in all samples

        if self.context.num_features is None:
            self.context.num_features = self.num_features
        else:
            assert self.context.num_features == self.num_features

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


class Model(object):
    def __init__(self, params, context=Context(), initialize=True):
        #print(params.shape, file=stderr)
        self.context = context
        self.params = np.array(params).T  # TODO: why pass transposed?
        self._params_sum = None
        self._params_log = None
        self._params_complement_log = None

        if context.num_features is None:
            context.num_features = self.num_features
        else:
            assert context.num_features == self.num_features

        if initialize:
            self.update()

    def update(self):
        self._params_log = np.log(self.params)
        self._params_complement_log = np.log(1. - self.params)
        return False  # indicates whether a dimension change occurred

    def update_context(self):  # TODO: implement proper context support
        pass

    def log_likelihood(self, data):  # TODO: check and adjust formula
        assert data.num_features == self.num_features
        term1 = np.dot(data.covmeans, self._params_log)  # TODO: scipy special.xlogy(k, p)?
        assert np.all(~np.isnan(term1))
        term2 = np.dot(data.covmeanstotal - data.covmeans, self._params_complement_log)  # TODO: scipy  special.xlog1py(n-k, -p)?
        assert np.all(~np.isnan(term2))
        loglike = term1 + term2 + data.conterm
        loglike = loglike/self.num_features  # normalize by number of samples
        common.write_probmatrix(loglike, file=logfile)
        assert np.all(loglike <= .0)
        return loglike

    def get_labels(self, indices=None):
        if not indices:
            indices = list(range(self.params.shape[1]))
        for i in indices:
            yield "-".join(("%i" % round(v) for v in np.asarray(self.params)[:, i]))

    def maximize_likelihood(self, responsibilities, data, cmask=None):  # TODO: adjust
        if cmask is None or cmask.shape == () or np.all(cmask):
            weights = responsibilities * data.sizes
        else:
            weights = responsibilities[:, cmask] * data.sizes

        weighted_meancoverage_samples = np.dot(data.covmeans.T, weights)
        weighted_meancoverage_total = np.dot(data.covmeanstotal.T, weights)

        self.params = (weighted_meancoverage_samples + 0.5) / (weighted_meancoverage_total + 0.5)  # introduced pseudocounts
        return self.update()

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


def empty_model(cluster_number, context, **kwargs):
    assert cluster_number > 0
    assert type(context) == Context
    params = np.zeros(shape=(cluster_number, context.num_features), dtype=common.prob_type)
    return Model(params, **kwargs)


def random_model(cluster_number, context, **kwargs):
    assert cluster_number > 0
    assert type(context) == Context
    params = np.random.rand(cluster_number, context.num_features)
    params /= params.sum(axis=1, keepdims=True)
    return Model(params, **kwargs)


def load_data_file(filename, **kwargs):
    d = Data(**kwargs)
    return common.load_data_file(filename, d)
