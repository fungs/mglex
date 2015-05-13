#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file holds all the functions and types necessary for probabilistic modelling of DNA composition.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import common
import numpy as np
from itertools import compress
from sys import stderr, exit


class Data(object):
    def __init__(self):
        self._frequencies = []
        self.sizes = None
        self.frequencies = None

    def deposit(self, frequencies):  # TODO: adjust signature with UniversalData
        frequencies = np.array(frequencies, dtype=self.composition_type)  # freqencies must not exceed 4,294,967,295
        maxtype = np.min_scalar_type(np.max(frequencies))
        row = frequencies.astype(maxtype)  # TODO: support sparse features
        self._frequencies.append(row)

    def prepare(self):
        self.frequencies = np.vstack(self._frequencies)
        self.sizes = self.frequencies.sum(axis=1, keepdims=True)
        print("frequencies before normalization")
        print(self.frequencies[:4, :])
        self.frequencies = self.frequencies / common.prob_type(self.sizes)
        print("frequencies after normalization")
        print(self.frequencies[:4, :])
        common.assert_probmatrix(self.frequencies)
        return self

    def parse(self, inseq):  # TODO: add load_data from generic with data-specific parse_line function
        for entry in inseq:
            self.deposit(entry.split(","))
        return self.prepare()

    @property
    def num_features(self):
        return self.frequencies.shape[1]

    def __len__(self):
        return self.frequencies.shape[0]

    composition_type = np.uint32


class Model(object):  # TODO: move names to supermodel
    def __init__(self, variables, names, initialize=True, pseudocount=False):  # TODO: add pseudocount implementation
        self.names = names
        self.variables = variables
        self._fmask = None
        self._loglikes = None
        self._pseudocount = pseudocount
        if initialize:
            self.update()

    def update(self):
        assert len(self.names) == self.variables.shape[0]  # TODO: check dimension param
        assert(self.variables.sum(axis=1).all())  # DEBUG: remove
        dimchange = False

        # reduction of model complexity
        if not self._pseudocount:
            self.variables = common.prob_type(self.variables / self.variables.sum(axis=1, keepdims=True))  # TODO: optimize memory use
            common.assert_probmatrix(self.variables)
            fmask_old = self._fmask
            self._fmask = np.asarray(self.variables, dtype=bool).all(axis=0)
            if fmask_old is not None and np.any(fmask_old != self._fmask):
                dimchange = True
                stderr.write("LOG %s: toggle features %s\n" % (self._short_name, " ".join(map(str, toggled_f))))
            self._loglikes = np.log(self.variables[:, self._fmask])
            # stderr.write("LOG %s: using %i features\n" % (self._short_name, self._fmask.sum()))
            return dimchange

        stderr.write("ERROR %s: pseudocount method not implemented\n" % self._short_name)
        exit(1)

        self.variables += 1  # TODO: change code
        self.variables = common.prob_type(self.variables / self.variables.sum(axis=0))  # TODO: optimize memory use
        common.assert_probmatrix(self.variables.T)
        self._loglikes = np.log(self.variables)
        return False

    def log_likelihood(self, data):
        # stderr.write("data dimension: %s, loglike dimension: %s\n" % (data.frequencies.shape, self._loglikes.shape))
        assert data.num_features == self._loglikes.shape[1]
        if self._pseudocount:
            stderr.write("ERROR %s: pseudocount method not implemented\n" % self._short_name)
            exit(1)
            return np.dot(data.frequencies, self._loglikes.T)  #/ common.prob_type(data.sizes.T)  # TODO: add to fmask version below
        return np.dot(data.frequencies[:, self._fmask], self._loglikes.T) #/ data.sizes  # DEBUG: last division term for normalization

    def maximize_likelihood(self, responsibilities, data, cmask=None):
        if cmask is not None:
            self.variables = np.dot(responsibilities[:, cmask].T, data.frequencies)
            self.names = list(compress(self.names, cmask))  # TODO: make self.names a numpy array?
        else:
            common.assert_probmatrix(data.frequencies)
            self.variables = np.dot(responsibilities.T, data.frequencies)
        print("updated model composition:")
        print(self.variables.shape)
        print(self.variables[:2, :])
        stderr.write("LOG M: Frequency sum: %.2f\n" % self.variables.sum())
        return self.update()

    @property
    def components(self):
        assert self.names is not None
        return len(self.names)

    @property
    def features_used(self):
        if self._pseudocount:
            return self.variables.shape[0]  # TODO: check dimension!
        assert self._fmask is not None
        return sum(self._fmask)

    _short_name = "NB_model"


def load_model_tuples(inseq, **kwargs):  # TODO: make generic
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
    return [Model(np.vstack(v), names, **kwargs) for v in cols]


# TODO: add load_data from generic with data-specific parse_line function
load_model = lambda i, **kwargs: load_model_tuples(common.parse_lines_comma(i), **kwargs)  # TODO: move to model class?


def random_model(component_number, feature_number):
    initial_freqs = np.asarray(np.random.rand(component_number, feature_number), dtype=common.prob_type)
    return Model(initial_freqs, list(map(str, list(range(component_number)))))


def empty_model(component_number, feature_number):
    initial_freqs = np.zeros(shape=(component_number, feature_number), dtype=common.prob_type)
    return Model(initial_freqs, list(map(str, list(range(component_number)))), initialize=False)
