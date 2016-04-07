u"""
This file holds all the functions and types necessary for probabilistic modelling of (read) coverage.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from .. import common, types
import numpy as np
from sys import stderr

mean_coverage_type = np.float32  # move to context object


class Context(object):
    """Container for information which is shared between Data and Model"""

    def __init__(self):
        self.num_features = None


class Data(object):
    def __init__(self, context=Context()):  # TODO: use deque() for large append-only lists
        self.context = context
        self.covmeans = None
        self.conterm = None
        self._covmeans = []

    def deposit(self, features):  # TODO: improve data parsing and handling
        coverage = np.array(features, dtype=mean_coverage_type)
        self._covmeans.append(coverage)

    def parse(self, inseq):  # TODO: add load_data from generic with data-specific parse_line function
        for entry in inseq:  # space-separated list of sample mean coverage
            self.deposit(entry.split(" "))
        return self.prepare()

    def prepare(self):
        self.covmeans = np.vstack(self._covmeans)
        self.conterm = common.gammaln(self.covmeans+1).sum(axis=1, keepdims=True)
        assert(np.all(self.covmeans.sum(axis=1) > 0))

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


class Model(object):
    def __init__(self, params, context=Context(), initialize=True, pseudocount=False):
        self.context = context

        if pseudocount:  # TODO: needs investigation, do not activate
            self.params = np.array(params + 1).T
            self._pseudocount = True
        else:
            self.params = np.array(params).T
            self._pseudocount = False

        self.stdev = None
        self._params_sum = None
        self._params_log = None

        if initialize:
            self.update()

    def update(self):
        # if not np.all(self.params):
            # print >>stderr, "some cluster in some sample wasn't observed:", self.params
        with np.errstate(divide='ignore'):
            self._params_sum = self.params.sum(axis=0, keepdims=True)
            self._params_log = np.log(self.params)
        return False  # indicates whether a dimension change occurred

    def update_context(self):  # TODO: implement proper context support
        pass

    def log_likelihood(self, data):
        # term1 = np.dot(data.covsums, self._params_log)  # sum of data coverage version
        term1 = np.dot(data.covmeans, self._params_log)  # mean coverage version
        # print("term1 shape is %ix%i" % term1.shape)
        # term2 = np.dot(data.sizes, self._params_sum)  # sum of data coverage version
        term2 = self._params_sum  # mean coverage version
        # print("term2 shape is %ix%i" % term2.shape)
        # loglike = term1 - term2
        # loglike = loglike - data.conterm  # optional if only scaled likelihood is needed
        loglike = np.asarray(term1 - term2 - data.conterm, dtype=types.logprob_type)/self.num_features
        # print >>stderr, loglike
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
        weights_normalization = weights_combined.sum(axis=0, keepdims=True)
        self.params = weighted_meancoverage_samples / weights_normalization

        dimchange = self.update()  # create cache for likelihood calculations

        ll = self.log_likelihood(data)

        # n=8000
        # test = ll[-n:, :]
        # common.print_probmatrix(ll[-n:, :])
        # print(np.sum(np.isinf(test)))

        # import sys
        # sys.exit(1)

        std_per_class = np.sqrt(common.weighted_variance(ll, weights_combined))
        weight_per_class = weights_combined.sum(axis=0, dtype=types.large_float_type)
        relative_weight_per_class = np.asarray(weight_per_class / weight_per_class.sum(), dtype=types.prob_type)
        combined_std = np.dot(std_per_class, relative_weight_per_class)
        # stderr.write("Weighted stdev was: %s\n" % common.pretty_probvector(std_per_class))
        # stderr.write("Weighted combined stdev was: %.2f\n" % combined_std)
        stderr.write("LOG %s: class likelihood standard deviation is %.2f\n" % (self._short_name, combined_std))
        self.stdev = combined_std
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

    # @property
    # def features_used(self):
        # return sum(self._fmask)

    _short_name = "PO_model"


def empty_model(cluster_number, context, **kwargs):
    assert cluster_number > 0
    assert type(context) == Context
    params = np.zeros(shape=(cluster_number, context.num_features), dtype=mean_coverage_type)
    return Model(params, **kwargs)


def random_model(cluster_number, context, low=0, high=None, **kwargs):
    assert cluster_number > 0
    assert type(context) == Context
    params = np.random.randint(low, high, (cluster_number, context.num_features))
    return Model(params, **kwargs)


def load_data_file(filename, **kwargs):
    d = Data(**kwargs)
    return common.load_data_file(filename, d)
