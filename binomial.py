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
    def __init__(self, sizes):
        self._covmeans = []  # TODO: use deque() for large append-only lists
        self.covmeans = None
        self.covmeanstotal = None
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
        assert data.num_features == self.num_features
        term1 = np.dot(data.covmeans, self._params_log)  # mean coverage version
        assert np.all(~np.isnan(term1))
        term2 = np.dot(data.covmeanstotal - data.covmeans, self._params_complement_log)  # mean coverage version
        assert np.all(~np.isnan(term2))
        loglike = term1 + term2
        loglike = loglike/self.num_features  # normalize by number of samples
        print_probmatrix(loglike, file=logfile)
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



def empty_model(component_number, feature_number):  # TODO: make generic
    params = np.zeros(shape=(component_number, feature_number), dtype=prob_type)
    return Model(params)


def random_model(component_number, feature_number):  # TODO: make generic
    params = np.random.rand(component_number, feature_number)
    params /= params.sum(axis=1, keepdims=True)
    return Model(params)
