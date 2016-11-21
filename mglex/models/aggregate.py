# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
This file holds all the functions and types necessary for the aggregate likelihood model.
"""

__author__ = "code@fungs.de"

from .. import common, types
import numpy as np
from sys import argv, exit, stdin, stdout, stderr, exit


class AggregateData(list):  # TODO: rename CompositeData
    def __init__(self, *args, **kwargs):
        super(AggregateData, self).__init__(*args, **kwargs)
        # try:
        #     self.sizes = np.asarray(sizes, dtype=self.size_type)
        # except TypeError:
        #     self.sizes = np.fromiter(sizes, dtype=self.size_type)[:, np.newaxis]

    def deposit(self, features):
        # self.names.append(name)
        for d, f in zip(self, features):
            # print(d, f)
            d.deposit(f)

    def prepare(self):
        for d in self:
            d.prepare()
        return self

    @property
    def num_data(self):
        if not super(AggregateData, self).__len__():
            return 0

        # print(self, file=stderr)
        num = self[0].num_data
        assert num == len(self[0])
        for l in self[1:]:
            assert(l.num_data == num)
        return num

    @property
    def num_features(self):
        return super(AggregateData, self).__len__()

    size_type = types.seqlen_type


class AggregateModel(list):  # TODO: rename CompositeModel, implement update() and maximize_likelihood()
    def __init__(self, *args, **kw):
        super(AggregateModel, self).__init__(*args, **kw)
        self.beta_correction = 1.0

    @property
    def names(self):
        if len(self) > 1:
            component_names = list(zip(*[m.names for m in self]))
            return [",".join(t) for t in component_names]
        return self[0].names

    @property
    def num_components(self):  # transitional
        if not len(self):
            return 0
        cluster_num = self[0].num_components
        assert np.all(np.equal(cluster_num, [model.num_components for model in self[1:]]))
        return cluster_num

    def log_likelihood(self, data):
        #assert self.weights.size == len(self)

        ll_scale = np.asarray([m.stdev if m.stdev > 0.0 else 0.1 for m in self])  # stdev of zero is not allowed, quick workaround!
        ll_weights = self.beta_correction*(ll_scale.sum()/ll_scale.size**2)/ll_scale
        ll_per_model = np.asarray([w*m.log_likelihood(d) for (m, d, w) in zip(self, data, ll_weights)])  # TODO: reduce memory usage, de-normalize scale

        s = np.mean(np.exp(ll_per_model), axis=1)  # TODO: remove debug calculations
        l = np.sum(ll_per_model, axis=1, dtype=types.large_float_type)  # TODO: remove debug calculations

        for m, mvec, lvec in zip(self, s, l):  # TODO: save memory
            stderr.write("LOG %s: average likelihood %s *** %s\n" % (m._short_name, common.pretty_probvector(mvec), common.pretty_probvector(lvec)))

        loglike = np.sum(ll_per_model, axis=0, keepdims=False)  # TODO: serialize
        return loglike

    def maximize_likelihood(self, data, responsibilities, weights, cmask=None):

        loglikelihood = np.zeros(shape=(data.num_data, self.num_components), dtype=types.logprob_type)
        return_value = False
        for m, d in zip(self, data):
            ret, ll = m.maximize_likelihood(d, responsibilities, weights, cmask)
            ll = loglikelihood + ll
            return_value = return_value and ret
        return return_value, loglikelihood