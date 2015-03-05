#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
Test EM algorithm for Poisson components with coverage.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import numpy as np
from numpy import log, exp, array, empty, repeat, finfo, zeros
from numpy.testing import assert_approx_equal
from numpy.random import randint
from itertools import izip, count
from sys import argv, exit, stdin, stdout, stderr, exit
from operator import itemgetter


def argmax(s, n=1):
    get_second = itemgetter(1)
    max_store = sorted(list(enumerate(s[:n])), key=get_second)
    for e in izip(count(n), s[n:]):
        max_store = sorted(max_store + [e], key=get_second)[-n:]  # presumably faster than theoretically better merging
    if n == 1:
        return max_store[0]
    return max_store


class FeatureHolder:
    def __init__(self, samples):
        self.samples = []
        self.seqnames = []
        self._samplename2index = {}
        self._covsums = []
        self._seqlens = []
        self._intialize_samples(samples)
        self._zero_count_vector = np.zeros(len(samples), dtype=np.int32)
        self.covsums = None
        self.seqlens = None

    def _intialize_samples(self, samples):
        for i, sample in enumerate(samples):
                self._samplename2index[sample] = i
                self.samples.append(sample)

    def deposit(self, name, feature_list):
        self.seqnames.append(name)
        row = self._zero_count_vector.copy()
        for sample, features in feature_list:
            try:
                index = self._samplename2index[sample]
                row[index] = np.sum(features)
                length = len(features)
                assert(row[index])
            except KeyError:
                stderr.write("Feature with sample name \"%s\" ignored.\n" % sample)
        self._covsums.append(row)
        self._seqlens.append(length)

    def prepare(self):
        self.covsums = np.mat(np.vstack(self._covsums))
        self.seqlens = np.mat(self._seqlens, dtype=np.int32)
        assert(np.all(self.covsums.sum(axis=1) > 0))
        return self


class PoissonComponent:
    def __init__(self, params):
        self.params = np.mat(params).T
        self._params_sum = self.params.sum(axis=0)
        self._params_log = log(self.params)

    def log_likelihood(self, data):
        #print (data.covsums * self._params_log).shape
        #stderr.write("sequence lengths: %s\nk_sums: %s\n" % (data.seqlens, self._params_sum))
        #stderr.write("summed data coverage:\n%s\nlog lambdas:\n%s\n" % (data.covsums, self._params_log))
        return (data.covsums * self._params_log) - (data.seqlens.T * self._params_sum)   # - data.cov_logfac

    def get_labels(self, indices=None):
        if not indices:
            indices = range(self.params.shape[1])
        for i in indices:
            yield "-".join(("%i" % round(v) for v in np.asarray(self.params)[:,i]))


def load_coverages(input):
    all_clists = []
    samples = input.next().rstrip().split("\t")
    for line in input:
        if not line or line[0] == "#":
            continue
        clist = line.rstrip().split("\t")
        if clist:
            all_clists.append(map(int, clist))
    return samples, all_clists


def log_normalize2(data):
    data = exp(data)
    data /= data.sum(axis=1)[:, None]
    return data


def log_normalize(data):
    data -= data.max(axis=1)  # avoid tiny numbers
    #threshold = log(finfo(data.dtype).resolution) - log(data.shape[1])  # TODO: adjust to float data type
    #select = data >= threshold
    #data[select] = exp(data[select])
    #data[not select] = 0.
    data = exp(data)
    data /= data.sum(axis=1)
    return data


def m_step(membership, data):
    #print membership
    assert_probmatrix(membership)
    priors = np.squeeze(np.asarray(data.seqlens*membership))
    priors /= priors.sum()
    assert_probarray(priors)
    weighted_coverage_sum = data.covsums.T * membership
    #print "weighted coverage sum:", weighted_coverage_sum
    weighted_length_sum = data.seqlens * membership
    #print "weighted length sum:", weighted_length_sum
    lambdas = (weighted_coverage_sum/weighted_length_sum).T
    priors, lambdas = remove_invalid(priors, lambdas)
    #print "parameters: ", lambdas.shape, lambdas.sum()
    #stderr.write("Finished maximization step\n\n")
    return PoissonComponent(lambdas), priors  # TODO: adjust transpose in PoissonComponent


def e_step(data, model, priors):
    membership = model.log_likelihood(data) + np.log(priors)
    total_mixture_loglike = membership.sum()
    stderr.write("total mixture likelihood: %.2f\n" % total_mixture_loglike)
    #stderr.write("log priors:\n%s\n\nlog membership with priors added:\n%s\n\n" % (log_priors, membership))
    membership = log_normalize(membership)
    #stderr.write("normalized membership:\n%s\n\n" % membership)
    #stderr.write("Finished expectation step\n")
    return membership


def parse_features(input, samples):
    store = FeatureHolder(samples)
    for line in input:
        if not line or line[0] == "#":  # skip empty lines and comments
            continue
        seqname, coverage_field = line.rstrip().split("\t", 2)[:2]
        feature_list = []
        for sample_group in coverage_field.split(" "):
            sample_name, coverage = sample_group.split(":", 2)[:2]
            coverage = map(int, coverage.split(","))  # TODO: use sparse numpy objects...
            feature_list.append((sample_name, coverage))
        store.deposit(seqname, feature_list)
    return store


def assert_probarray(v):
    assert_approx_equal(v.sum(), 1.)


def assert_probmatrix(membership):
    is_sum = membership.sum()
    should_sum = membership.shape[0]
    assert_approx_equal(is_sum, should_sum)


def remove_invalid(priors, lambdas):
    old_shape = lambdas.shape
    index = np.where(priors)[0]  # TODO: do it right
    lambdas = lambdas[index, :]
    if lambdas.shape != old_shape:
        stderr.write("shrinked number of cluster from %i to %i\n" % (old_shape[0], lambdas.shape[0]))
    return priors[index], lambdas


def pretty_print(lambdas, priors, output=stderr):
    for l, p in izip(np.asarray(lambdas), priors):
        output.write("lambdas: [%s], prior: %.2f\n" % (",".join(("% 5.1f" % v for v in l)), p))


def approx_equal(v1, v2, precision):
    if v1.shape != v2.shape:
        return False
    return ((v1-v2) < precision).all()


if __name__ == "__main__":
    # command line options and constants
    samples = argv[1:]
    stderr.write("parsing features\n")
    data = parse_features(stdin, samples)
    data.prepare()
    C = len(samples)
    
    stderr.write("samples: %s\n" % ", ".join(data.samples))

    # initialize clusters
    #lambdas = array(((2., 16., 31., 8.)[:C], (2., 6., 9., 6.)[:C], (2., 7., 3., 11.)[:C], (3., 6., 14., 46)[:C], (4., 15., 1., 1.)[:C],
    #                 (5., 18., 9., 3.)[:C], (5., 9., 1., 3.)[:C], (11., 10., 1., 5.)[:C], (20., 11., 18., 2.)[:C], (46., 2., 13., 15.)[:C]))
    #lambdas = array(((2., randint(1, 50), randint(1, 50), randint(1, 50))[:C],
    #                 (2., randint(1, 50), randint(1, 50), randint(1, 50))[:C],
    #                 (2., randint(1, 50), randint(1, 50), randint(1, 50))[:C],
    #                 (3., randint(1, 50), randint(1, 50), randint(1, 50))[:C],
    #                 (4., randint(1, 50), randint(1, 50), randint(1, 50))[:C],
    #                 (5., randint(1, 50), randint(1, 50), randint(1, 50))[:C],
    #                 (5., randint(1, 50), randint(1, 50), randint(1, 50))[:C],
    #                 (11., randint(1, 50), randint(1, 50), randint(1, 50))[:C],
    #                 (20., randint(1, 50), randint(1, 50), randint(1, 50))[:C],
    #                 (46., randint(1, 50), randint(1, 50), randint(1, 50))[:C]))
    lambdas = array(((randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50))[:C],
                     (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50))[:C],
                     (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50))[:C],
                     (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50))[:C],
                     (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50))[:C],
                     (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50))[:C],
                     (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50))[:C],
                     (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50))[:C],
                     (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50))[:C],
                     (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50))[:C]))
    priors = repeat(1./lambdas.shape[0], lambdas.shape[0])  # evenly distributed (flat) priors
    #lambdas_last = zeros(lambdas.size).reshape(lambdas.shape)
    priors_last = zeros(priors.size)
    cm = PoissonComponent(lambdas)

    # EM iterations TODO: both steps at once (summary statistics)
    iteration_step = count(1)
    while not (approx_equal(priors, priors_last, precision=.001) and approx_equal(cm.params, cm_last.params, precision=.001)) :  # continue until convergence
        stderr.write("iteration step %i\n" % iteration_step.next())
        assert_probarray(priors)
        pretty_print(cm.params.T, priors)
        membership = e_step(data, cm, priors)
        assert_probmatrix(membership)
        priors_last, cm_last = priors, cm
        cm, priors = m_step(membership, data)
        stderr.write("\n")

    # output clusters with highest membership value
    cluster_names = ["-".join(("%i" % round(v) for v in params)) for params in lambdas]
    for name, mem in izip(data.seqnames, membership):
        cluster_index, cluster_prob = argmax(mem)
        stdout.write("%s\t%s\t%.2f\n" % (name, cluster_names[cluster_index], cluster_prob))

