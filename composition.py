#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
This file holds all the functions and types necessary for probabilistic modelling of DNA composition.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import common
import numpy as np
from itertools import compress
from sys import stderr


freq_type = np.uint16


class Data(object):
    def __init__(self):
        # self.names = []
        self._frequencies = []
        self.sizes = None
        self.frequencies = None

    def deposit(self, frequencies):  # TODO: adjust signature with UniversalData
        # self.names.append(name)
        frequencies = np.array(frequencies, dtype=np.uint32)  # freqencies must not exceed 4,294,967,295
        maxtype = np.min_scalar_type(np.max(frequencies))
        row = frequencies.astype(maxtype)  # TODO: support sparse features
        self._frequencies.append(row)

    def prepare(self):
        self.frequencies = np.mat(np.vstack(self._frequencies))  # TODO: check variable type casting behavior
        # tmp = self.frequencies.sum(axis=1)
        # self.frequencies /= np.float32(tmp)  # TODO: normalization necessary? make smarter operations, calculate earlier
        # self.sizes = tmp.T
        self.sizes = self.frequencies.sum(axis=1).T
	self.frequencies = self.frequencies / common.prob_type(self.sizes.T)
        return self

    @property
    def num_features(self):
        return self.frequencies.shape[1]

    def __len__(self):
        return self.frequencies.shape[0]


class Model(object):  # TODO: move names to supermodel
    def __init__(self, variables, names, initialize=True):
        self.variables = np.mat(variables).T
        # print variables.shape, "->", self.variables.shape
        self.names = names
        self._fmask = None
        if initialize:
            self.update()

    def update(self):
        assert len(self.names) == self.variables.shape[1]  # TODO: check dimension param
        dimchange = False
        assert(self.variables.sum(axis=0).all())  # DEBUG: remove
        self.variables = common.prob_type(self.variables / self.variables.sum(axis=0))  # TODO: optimize memory use
        # assert_probmatrix(self.variables.T)

        # reduction of model complexity
        fmask_old = self._fmask
        self._fmask = np.asarray(self.variables, dtype=bool).all(axis=1)
        if fmask_old is not None and np.any(fmask_old != self._fmask):
            dimchange = True
            # toggled_f = np.where(self._fmask != fmask_old)[0]
            # stderr.write("LOG %s: toggle features %s\n" % (self._short_name, " ".join(map(str, toggled_f))))
            stderr.write("LOG %s: using %i features\n" % (self._short_name, self._fmask.sum()))

        # self._loglikes = np.log(self.variables)
        self._loglikes = np.log(self.variables[self._fmask])
        return dimchange

    def log_likelihood(self, data):
#	print >> stderr, "data dimension: %s, loglike dimension: %s" % (data.frequencies.shape, self._loglikes.shape)
	assert data.num_features == self._loglikes.shape[0]
        return data.frequencies[:, self._fmask] * self._loglikes #/ data.sizes  # DEBUG: last division term for normalization
        #return data.frequencies * self._loglikes

    def maximize_likelihood(self, responsibilities, data, cmask=None):
        if cmask is not None:
            self.variables = data.frequencies.T * responsibilities[:, cmask]
            self.names = list(compress(self.names, cmask))  # TODO: make self.names a numpy array?
        else:
            self.variables = data.frequencies.T * responsibilities
        # stderr.write("LOG M: Frequency sum: %.2f\n" % self.variables.sum())
        return self.update()

    @property
    def components(self):
        assert self.names is not None
        return len(self.names)

    @property
    def features_used(self):
        assert self._fmask is not None
        return sum(self._fmask)

    _short_name = "NB_model"


def load_model_tuples(inseq):  # TODO: make generic
    cols = []
    names = []
    try:
        for rec in inseq:
            names.append(rec[0])
            for i, data in enumerate(rec[1:]):
                vec = np.asarray(data, dtype=np.float64)  # allow for large numbers
                try:
                    cols[i].append(vec)
                except IndexError:
                    cols.append([vec])
    except TypeError:
        stderr.write("Could not parse model definition line\n")
        exit(1)
    return map(lambda v: Model(np.vstack(v), names), cols)


# TODO: add load_data from generic with data-specific parse_line function
load_model = lambda i: load_model_tuples(common.parse_lines(i))  # TODO: move to model class?


def random_model(component_number, feature_number):
    initial_freqs = np.asarray(np.random.rand(component_number, feature_number), dtype=common.prob_type)
    return Model(initial_freqs, map(str, range(component_number)))


def empty_model(component_number, feature_number):
    initial_freqs = np.zeros(shape=(component_number, feature_number), dtype=common.prob_type)
    return Model(initial_freqs, map(str, range(component_number)), initialize=False)
