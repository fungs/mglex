# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
This file holds all the functions and types necessary for probabilistic modelling of DNA composition.
"""

__author__ = "code@fungs.de"

from .. import common, types
import numpy as np
from itertools import compress
from sys import stderr, exit


class Context(object):
    """Container for information which is shared between Data and Model"""

    def __init__(self):
        self.num_features = None


class Data(object):
    def __init__(self, context=Context()):
        self.context = context
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
        self.sizes = self.frequencies.sum(axis=1, keepdims=True, dtype=types.seqlen_type)  # TODO: replace by global seqlen
        # print("Data frequencies", file=logfile)
        # common.print_vector(self.frequencies[0, :], file=logfile)  # TODO: remove
        # common.print_vector(self.frequencies[-1, :], file=logfile)  # TODO: remove
        # common.newline(file=logfile)
        self.frequencies = types.prob_type(self.frequencies/self.sizes)  # TODO: why does /= not work?
        common.assert_probmatrix(self.frequencies)

        self.context.num_features = self.num_features
        return self

    def parse(self, inseq):  # TODO: add load_data from generic with data-specific parse_line function
        for entry in inseq:
            self.deposit(entry.split(" "))
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
    def __init__(self, variables, names, context=Context(), initialize=True, pseudocount=False):  # TODO: add pseudocount implementation
        self.context = context
        self.names = names
        self.variables = variables
        self.stdev = None
        self._fmask = None
        self._loglikes = None
        self._pseudocount = pseudocount

        if context.num_features is None:
            context.num_features = self.num_features
        else:
            assert context.num_features == self.num_features

        if initialize:
            self.variables = types.prob_type(self.variables/self.variables.sum(axis=1, keepdims=True))  # normalize
            self.update()

    def update(self):
        assert len(self.names) == self.variables.shape[0]  # TODO: check dimension param
        common.assert_probmatrix_relaxed(self.variables)
        dimchange = False

        # reduction of model complexity
        if not self._pseudocount:
            fmask_old = self._fmask
            self._fmask = np.all(self.variables, axis=0)
            if fmask_old is not None and np.any(fmask_old != self._fmask):  # TODO: consider dimchange when fmask_old==None
                dimchange = True
                #stderr.write("LOG %s: toggle features %s\n" % (self._short_name, " ".join(map(str, toggled_f))))
            self._loglikes = np.log(self.variables[:, self._fmask])
            stderr.write("LOG %s: using %i out of %i features\n" % (self._short_name, self._fmask.sum(), self.variables.shape[1]))

            # print("Model composition for %i clusters and %i features:" % self.variables.shape, file=logfile)
            # common.print_probvector(self.variables[0, :], file=logfile)
            # common.print_probvector(self.variables[-1, :], file=logfile)
            # common.newline(file=logfile)
            return dimchange

        stderr.write("ERROR %s: pseudocount method not implemented\n" % self._short_name)
        exit(1)

        # simple pseudocount method (add frequency 1 to all counts), TODO: maybe add .5 where necessary only, or leave to initialization method
        # problem: update() is called after each maximize step

        self.variables += 1  # TODO: change code
        self.variables = types.prob_type(self.variables/self.variables.sum(axis=1, keepdims=True))  # TODO: optimize memory usage
        common.assert_probmatrix(self.variables)
        self._loglikes = np.log(self.variables)
        return False

    def update_context(self):  # TODO: implement proper context support
        pass

    def log_likelihood(self, data):
        # stderr.write("data dimension: %s, loglike dimension: %s\n" % (data.frequencies.shape, self._loglikes.shape))
        assert data.num_features == self.num_features  # TODO: do not test with every call, instead check context equality?
        if self._pseudocount:
            stderr.write("ERROR %s: pseudocount method not implemented\n" % self._short_name)
            exit(1)
            loglike = np.dot(data.frequencies, self._loglikes.T)  #/ common.prob_type(data.sizes.T)  # TODO: add to fmask version below
        else:
            loglike = np.dot(data.frequencies[:, self._fmask], self._loglikes.T) #/ data.sizes  # DEBUG: last division term for normalization

        assert np.all(np.logical_or(np.isnan(loglike), loglike <= 0.0))
        return loglike

    def maximize_likelihood(self, data, responsibilities, weights, cmask=None):

        if not (cmask is None or cmask.shape == () or np.all(cmask)):  # cluster reduction
            responsibilities = responsibilities[:, cmask]
            self.names = list(compress(self.names, cmask))  # TODO: make self.names a numpy array?

        weights_combined = responsibilities * weights

        self.variables = np.dot(weights_combined.T, data.frequencies)
        with np.errstate(invalid='ignore'):  # if no training data is available for any class
            np.divide(self.variables, weights_combined.sum(axis=0, keepdims=True, dtype=types.large_float_type).T, out=self.variables)  # normalize before update, self.variables is types.prob_type

        dimchange = self.update()  # create cache for likelihood calculations

        # TODO: refactor this block
        ll = self.log_likelihood(data)
        std_per_class = common.weighted_std(ll, weights_combined)
        weight_per_class = weights_combined.sum(axis=0, dtype=types.large_float_type)
        weight_per_class /= weight_per_class.sum()
        std_per_class_mask = np.isnan(std_per_class)
        skipped_classes = std_per_class_mask.sum()
        self.stdev = np.ma.dot(np.ma.MaskedArray(std_per_class, mask=std_per_class_mask), weight_per_class)
        stderr.write("LOG %s: mean class likelihood standard deviation is %.2f (omitted %i/%i classes due to invalid or unsufficient data)\n" % (self._short_name, self.stdev, skipped_classes, self.num_components))
        return dimchange, ll

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

    @property
    def num_features(self):
        return self.variables.shape[1]  # TODO: check dimension!

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

def random_model(cluster_number, context, **kwargs):
    assert cluster_number > 0
    assert type(context) == Context
    initial_freqs = np.asarray(np.random.rand(cluster_number, context.num_features), dtype=types.prob_type)
    return Model(initial_freqs, list(map(str, list(range(cluster_number)))), **kwargs)


def empty_model(cluster_number, context, **kwargs):
    assert cluster_number > 0
    assert type(context) == Context
    initial_freqs = np.ones(shape=(cluster_number, context.num_features), dtype=types.prob_type)
    return Model(initial_freqs, list(map(str, list(range(cluster_number)))), initialize=False, **kwargs)


def load_data_file(filename, **kwargs):
    d = Data(**kwargs)
    return common.load_data_file(filename, d)
