#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test EM algorithm for Poisson components with coverage.
"""

from scipy.stats import poisson
from numpy import log, exp, array, empty, repeat, finfo, zeros
from numpy.testing import assert_approx_equal
from numpy.random import randint
from itertools import izip, count
from collections import Counter
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
        self._samplename2index = {}
        self._sample_columns = []
        self.count = 0
        self.samples = []
        self.seqnames = []
        self.sequence_length = []
        self._intialize_samples(samples)

    def _intialize_samples(self, samples):
        for i, sample in enumerate(samples):
                self._samplename2index[sample] = i
                self._sample_columns.append([])
                self.samples.append(sample)

    def deposit(self, name, feature_list):
        self.seqnames.append(name)
        for sample, features in feature_list:
            feat = Counter(features).items()
            if len(self.sequence_length) == self.count:
                self.sequence_length.append(len(features))
            try:
                index = self._samplename2index[sample]
                col = self._sample_columns[index]
                colsize = len(col)
                while colsize < self.count:
                    col.append(((0, self.sequence_length[colsize]),))
                    colsize += 1
                col.append(feat)
            except KeyError:
                stderr.write("Feature with sample name \"%s\" ignored.\n" % sample)
        self.count += 1


class PoissonComponent:
    def __init__(self, lambdas, max_coverage=1000):
        self.lambdas = tuple(lambdas)
        self.dists = tuple(poisson(l) for l in lambdas)

    def log_likelihood(self, features_list):
        total = .0
        for i, features in enumerate(features_list):
            for k, n in features:
                total += n*self._log_likelihood_single(i, k)
        return total

    def likelihood(self, features_list):
        total = .0
        for i, features in enumerate(features_list):
            for k, n in features:
                total *= n*self._likelihood_single(i, k)
        return total

    def get_label(self):
        return "-".join(map(str, self.lambdas))

    def _log_likelihood_single(self, i, k):
        return self.dists[i].logpmf(k)

    def _likelihood_single(self, i, k):
        return self.dists[i].pmf(k)


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
    #threshold = log(finfo(data.dtype).resolution) - log(data.shape[1])  # TODO: adjust to float data type
    data -= data.max(axis=1)[:, None]
    #print data
    data = exp(data)
    #data[data >= threshold] = exp(data[data >= threshold])
    #data[data < threshold] = 0.
    data /= data.sum(axis=1)[:, None]
    return data


def m_step(membership, data):
    lambdas = empty((membership.shape[1], len(data.samples)))  # TODO: re-use field (or copy)
    stderr.write("available samples: %s, %i\n" % (data.samples, len(data._sample_columns)))
    for cluster_index, weights in enumerate(membership.T):
        for sample_index, features in enumerate(data._sample_columns):
            #stderr.write("processing data size %i; cluster %i; sample %i\n" % (len(weights), cluster_index, sample_index))
            coverages, lengths = zip(*(map(sum, zip(*((n*cov, n) for cov, n in feat))) for feat in features))  # TODO: avoid calc lengths
            #print sample_index, coverages, lengths, coverages/float(lengths)
            #for w, c, l in izip(weights, coverages, lengths):
                #stderr.write("weight*coverage: %.2f*%.2f=%.2f | weight*length: %.2f*%.2f=%.2f\n" % (w, c, w*c, w, l, w*l))
            lambdas[cluster_index][sample_index] = (weights*coverages).sum()/(weights*lengths).sum()
    priors = (membership.T*lengths).sum(axis=1)
    priors /= priors.sum()
    stderr.write("Finished maximization step\n\n")
    return lambdas, priors


def e_step(data, classes, priors):
    log_priors = log(priors)
    #print log_priors
    membership = empty((data.count, len(classes)))  # TODO: re-use field (or copy)
    for i, d in enumerate(izip(*data._sample_columns)):
        membership[i] = map(lambda c: c.log_likelihood(d), classes) + log_priors  # weight by priors TODO: without_loop
        #stderr.write("sequence %i: %s\n" % (i,membership[i]))
    # weight and normalize
    #print membership
    membership = log_normalize(membership)
    #for i, mem in enumerate(membership):
        #stderr.write("sequence %i: %s\n" % (i,mem))
        #assert_probarray(mem)
        #stderr.write("sequence membership array: %s, sum of probs: %.2f\n" % (mem, exp(mem).sum()))
    stderr.write("Finished expectation step\n\n")
    return membership


def parse_features(input, samples):
    store = FeatureHolder(samples)
    #seqnames = []
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
    index = array(priors, dtype=bool)
    return priors[index], lambdas[index]


def pretty_print(lambdas, priors, output=stderr):
    for l, p in izip(lambdas, priors):
        output.write("lambdas: %s, prior: %.2f\n" % (l, p))
    output.write("\n")


def approx_equal(v1, v2, precision=.001):
    if v1.shape != v2.shape:
        return False
    return ((v1-v2) < precision).all()


if __name__ == "__main__":
    # command line options and constants
    samples = argv[1:]
    stderr.write("parsing features\n")
    data = parse_features(stdin, samples)
    C = len(samples)
    
    stderr.write("samples: %s" % ", ".join(data.samples))

    # initialize clusters
    lambdas = array(((2., 16., 31., 8.)[:C], (2., 6., 9., 6.)[:C], (2., 7., 3., 11.)[:C], (3., 6., 14., 46)[:C], (4., 15., 1., 1.)[:C],
                     (5., 18., 9., 3.)[:C], (5., 9., 1., 3.)[:C], (11., 10., 1., 5.)[:C], (20., 11., 18., 2.)[:C], (46., 2., 13., 15.)[:C]))
    #lambdas = array(((2., randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (2., randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (2., randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (3., randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (4., randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (5., randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (5., randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (11., randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (20., randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (46., randint(1, 50), randint(1, 50), randint(1, 50))))
    #lambdas = array(((randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50)),
    #                 (randint(1, 50), randint(1, 50), randint(1, 50), randint(1, 50))))
    priors = repeat(1./lambdas.shape[0], lambdas.shape[0])  # evenly distributed (flat) priors
    lambdas_last = zeros(lambdas.size).reshape(lambdas.shape)
    priors_last = zeros(priors.size)

    # EM iterations TODO: both steps at once (summary statistics)
    iteration_step = count(1)
    while not approx_equal(priors, priors_last):  # continue until convergence
    #for i in range(2):
        stderr.write("iteration step %i\n" % iteration_step.next())
        assert_probarray(priors)
        classes = tuple(PoissonComponent(l) for l in lambdas)
        pretty_print(lambdas, priors)
        membership = e_step(data, classes, priors)
        assert_probmatrix(membership)
        #lambdas_last = lambdas
        priors_last = priors
        lambdas, priors = m_step(membership, data)
        priors, lambdas = remove_invalid(priors, lambdas)
    pretty_print(lambdas, priors)

    cluster_names = ["-".join(("%i" % round(v) for v in params)) for params in lambdas]
    for name, mem in izip(data.seqnames, membership):
        cluster_index, cluster_prob = argmax(mem)
        stdout.write("%s\t%s\t%.2f\n" % (name, cluster_names[cluster_index], cluster_prob))
