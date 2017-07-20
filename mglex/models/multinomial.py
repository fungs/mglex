# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
 This file holds all the functions and types necessary for probabilistic modelling of differential (read) coverage.
 We use a Multinomial PMF to model the relative coverage per position which also handles low count values.
"""

__author__ = "code@fungs.de"

from .. import common, types
import numpy as np
from sys import argv, exit, stdin, stdout, stderr, exit

# count data type
frequency_type = np.int32


class Context(object):
    """Container for information which is shared between Data and Model"""

    def __init__(self):
        self.num_features = None


class Data(object):
    def __init__(self, context=Context()):
        self.context = context
        self._covmeans = []  # TODO: use deque() for large append-only lists
        self.covmeans = None
        self.covmeanstotal = None
        self.conterm = None

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
        self.conterm = np.asarray(common.logmultinom(self.covmeanstotal, self.covmeans)/self.num_features, dtype=self.mean_coverage_type)  # shrink to prevent type overflow
        self.covmeans /= self.num_features  # scale numbers, avoid division during likelihood calculation
        self.covmeanstotal /= self.num_features  # scale numbers, avoid division during likelihood calculation
        
        assert np.all(np.isfinite(self.conterm))  # variable overflow in multinomial coefficient
        # assert(np.all(self.covmeanstotal > 0))  # TODO: zero observations are valid

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
        self.stdev = None
        self._params_log = None

        if context.num_features is None:
            context.num_features = self.num_features
        else:
            assert context.num_features == self.num_features

        if initialize:
            self.update()

    def update(self):
        with np.errstate(divide='ignore'):
            self._params_log = np.asarray(np.log(self.params), dtype=types.logprob_type)
        return False  # indicates whether a dimension change occurred

    def update_context(self):  # TODO: implement proper context support
        pass

    def log_likelihood(self, data):
        assert data.num_features == self.num_features

        term1 = common.nandot(data.covmeans, self._params_log)  # TODO: scipy special.xlogy(k, p)?
        assert np.all(~np.isnan(term1))
        
        loglike = np.asarray(term1 + data.conterm, dtype=types.logprob_type)

        loglike_is_negative = np.all(loglike <= .0)
        if not loglike_is_negative:
            mask = np.where(loglike > .0)
            for row, col, val in zip(mask[0], mask[1], loglike[mask]):
                stderr.write("LOG %s: positive loglikelihood violation row %i, col %i, val %.2f\n" % (self._short_name, row, col, val))
        assert loglike_is_negative  # non-negative log-likelihood is most likely due to variable precision error
        return loglike

    def get_labels(self, indices=None):
        if not indices:
            indices = list(range(self.params.shape[1]))
        for i in indices:
            yield "-".join(("%i" % round(v) for v in np.asarray(self.params)[:, i]))

    def maximize_likelihood(self, data, responsibilities, weights, cmask=None):

        if not (cmask is None or cmask.shape == () or np.all(cmask)):  # cluster reduction
            responsibilities = responsibilities[:, cmask]

        weights_combined = responsibilities * weights

        weighted_meancoverage_samples = np.dot(data.covmeans.T, weights_combined)  # TODO: use np.average?
        weighted_meancoverage_total = np.dot(data.covmeanstotal.T, weights_combined)  # TODO: use np.average? simplify?
        #weighted_meancoverage_total[weighted_meancoverage_total==.0] = np.nan  # no data -> undefined params
        assert np.all(weighted_meancoverage_total > 0.)  # zero coverage bins over given samples not allowed!

        pseudocount = 0.0000000001  # TODO: refine
        self.params = np.asarray((weighted_meancoverage_samples + pseudocount) / (weighted_meancoverage_total + pseudocount),
                                 dtype=types.prob_type)  # introduced pseudocounts

        dimchange = self.update()  # create cache for likelihood calculations

        # TODO: refactor this block
        ll = self.log_likelihood(data)
        std_per_class = common.weighted_std(ll, weights_combined)
        weight_per_class = weights_combined.sum(axis=0, dtype=types.large_float_type)
        weight_per_class /= weight_per_class.sum()
        std_per_class_mask = np.isnan(std_per_class)
        skipped_classes = std_per_class_mask.sum()
        self.stdev = np.ma.dot(np.ma.MaskedArray(std_per_class, mask=std_per_class_mask), weight_per_class)
        stderr.write("LOG %s: mean class likelihood standard deviation is %.2f (omitted %i/%i classes due to invalid or unsufficient data)\n" % (self._short_name, self.stdev, skipped_classes, self.num_components - skipped_classes))
        return dimchange, ll

    @property
    def num_components(self):
        return self.params.shape[1]

    @property
    def num_features(self):
        return self.params.shape[0]

    @property
    def names(self):
        return list(self.get_labels())

    _short_name = "MI_model"


def empty_model(cluster_number, context, **kwargs):
    assert cluster_number > 0
    assert type(context) == Context
    params = np.zeros(shape=(cluster_number, context.num_features), dtype=types.prob_type)
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
