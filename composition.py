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

logfile = open("coverage.log", "w")


class Data(object):
    def __init__(self):
        self._frequencies = []  # TODO: use deque() for large append-only lists
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
        print("Data frequencies", file=logfile)
        common.print_vector(self.frequencies[0, :], file=logfile)
        common.print_vector(self.frequencies[-1, :], file=logfile)
        common.newline(file=logfile)
        self.frequencies = self.frequencies / common.prob_type(self.sizes)  # TODO: why does /= not work?
        # print("data frequencies after normalization")
        # common.print_probvector(self.frequencies[0, :])
        # common.print_probvector(self.frequencies[-1, :])
        # common.newline()
        common.assert_probmatrix(self.frequencies)
        return self

    def parse(self, inseq):  # TODO: add load_data from generic with data-specific parse_line function
        for entry in inseq:
            self.deposit(entry.split(","))
        return self.prepare()

    @property
    def num_features(self):
        return self.frequencies.shape[1]

    @property
    def num_data(self):
        return self.frequencies.shape[0]

    def __len__(self):
        return self.num_data  # TODO: select an intuitive convention for this

    composition_type = np.uint32


class Model(object):  # TODO: move names to supermodel
    def __init__(self, variables, names, initialize=True, pseudocount=False):  # TODO: add pseudocount implementation
        self.names = names
        self.variables = variables
        self._fmask = None
        self._loglikes = None
        self._pseudocount = pseudocount

        if initialize:
            self.variables = common.prob_type(self.variables / self.variables.sum(axis=1, keepdims=True))  # normalize
            self.update()

    def update(self):
        assert len(self.names) == self.variables.shape[0]  # TODO: check dimension param
        common.assert_probmatrix(self.variables)
        dimchange = False

        # reduction of model complexity
        if not self._pseudocount:
            fmask_old = self._fmask
            self._fmask = np.asarray(self.variables, dtype=bool).all(axis=0)
            if fmask_old is not None and np.any(fmask_old != self._fmask):  # TODO: consider dimchange when fmask_old==None
                dimchange = True
                #stderr.write("LOG %s: toggle features %s\n" % (self._short_name, " ".join(map(str, toggled_f))))
            self._loglikes = np.log(self.variables[:, self._fmask])
            stderr.write("LOG %s: using %i out of %i features\n" % (self._short_name, self._fmask.sum(), self.variables.shape[1]))

            print("Model composition for %i clusters and %i features:" % self.variables.shape, file=logfile)
            common.print_probvector(self.variables[0, :], file=logfile)
            common.print_probvector(self.variables[-1, :], file=logfile)
            common.newline(file=logfile)
            return dimchange

        stderr.write("ERROR %s: pseudocount method not implemented\n" % self._short_name)
        exit(1)

        # simple pseudocount method (add frequency 1 to all counts), TODO: maybe add .5 where necessary only, or leave to initialization method
        # problem: update() is called after each maximize step

        self.variables += 1  # TODO: change code
        self.variables = common.prob_type(self.variables / self.variables.sum(axis=1, keepdims=True))  # TODO: optimize memory usage
        common.assert_probmatrix(self.variables)
        self._loglikes = np.log(self.variables)
        return False

    def log_likelihood(self, data):
        # stderr.write("data dimension: %s, loglike dimension: %s\n" % (data.frequencies.shape, self._loglikes.shape))
        assert data.num_features == self.variables.shape[1]
        if self._pseudocount:
            stderr.write("ERROR %s: pseudocount method not implemented\n" % self._short_name)
            exit(1)
            loglike = np.dot(data.frequencies, self._loglikes.T)  #/ common.prob_type(data.sizes.T)  # TODO: add to fmask version below
        else:
            loglike = np.dot(data.frequencies[:, self._fmask], self._loglikes.T) #/ data.sizes  # DEBUG: last division term for normalization
        assert np.all(loglike < .0)
        return loglike

    def maximize_likelihood(self, responsibilities, data, cmask=None):

        common.assert_probmatrix(data.frequencies)  # TODO: remove
        common.assert_probmatrix(responsibilities)  # TODO: remove

        if cmask is not None:
            responsibilities = responsibilities[:, cmask]
            self.names = list(compress(self.names, cmask))  # TODO: make self.names a numpy array?

        weights = responsibilities*data.sizes
        #assert weights.shape == responsibilities.shape

        self.variables = np.dot(weights.T, data.frequencies)  # TODO: consider data types (becomes float64?)
        self.variables = common.prob_type(self.variables/weights.sum(axis=0, keepdims=True).T)  # normalize before update

        # print("maximum likelihood model composition for %i clusters and %i features:" % self.variables.shape)
        # common.print_probvector(self.variables.sum(axis=1))
        # common.print_vector(responsibilities.sum(axis=0))
        # common.print_probvector(self.variables[0, :])
        # common.print_probvector(self.variables[-1, :])
        # common.newline()
        # stderr.write("LOG M: Frequency sum: %.2f\n" % self.variables.sum())
        return self.update()  # TODO: fix double normalization in update()

    @property
    def num_components(self):
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


def random_model(component_number, feature_number, pseudocount=False):
    initial_freqs = np.asarray(np.random.rand(component_number, feature_number), dtype=common.prob_type, pseudocount=pseudocount)
    return Model(initial_freqs, list(map(str, list(range(component_number)))))


def empty_model(component_number, feature_number, pseudocount=False):
    initial_freqs = np.ones(shape=(component_number, feature_number), dtype=common.prob_type)
    return Model(initial_freqs, list(map(str, list(range(component_number)))), initialize=False, pseudocount=pseudocount)
