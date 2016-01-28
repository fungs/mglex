#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test classification to Poisson components with coverage.
"""

from scipy.stats import poisson, chi2
from numpy import log, exp
from operator import itemgetter

get_second = itemgetter(1)


def argmax_two(s):
    max1 = (None, None)
    max2 = (None, None)
    for e in enumerate(s):
        max2, max1 = sorted((max1, max2, e), key=get_second)[-2:]
    return max1[0], max2[0]

get_significance = lambda l1, l2: 1. - chi2.cdf(-2*(l1 - l2), 1)


class FeatureHolder:
    def __init__(self, samples):
        self._samplename2index = {}
        self._sample_columns = []
        self.count = 0
        self.samples = []
        self.sequence_length = []
        self._intialize_samples(samples)

    def _intialize_samples(self, samples):
        for i, sample in enumerate(samples):
                self._samplename2index[sample] = i
                self._sample_columns.append([])
                self.samples.append(sample)

    def deposit(self, feature_list):
        for sample, features in feature_list:
            feat = Counter(features).items()
            if len(self.sequence_length) == self.count:
                self.sequence_length.append(len(features))
            index = self._samplename2index[sample]
            col = self._sample_columns[index]
            colsize = len(col)
            while colsize < self.count:
                col.append(((0, self.sequence_length[colsize]),))
                colsize += 1
            col.append(feat)
        self.count += 1


class PoissonComponent:
    def __init__(self, lambdas):
        self.lambdas = tuple(lambdas)
        self.dists = tuple(poisson(l) for l in lambdas)

    def log_likelihood(self, features_list):
        total = .0
        for i, features in enumerate(features_list):
            for k, n in features:
                #print "%i*self._log_likelihood_single(%i, %i)" % (n, i, k)
                total += n*self._log_likelihood_single(i, k)
        return total

    def get_label(self):
        return "-".join(map(str, self.lambdas))

    def _log_likelihood_single(self, i, k):
        return self.dists[i].logpmf(k)


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


if __name__ == "__main__":
    from sys import argv, exit, stdin, stdout
    from collections import Counter
    from itertools import izip

    # constants
    coverage_file = argv[1]

    # remember sequence names
    seqnames = []

    # parse coverages
    samples, coverages = load_coverages(open(coverage_file, "r"))

    # parse features from stdin
    features_per_sample = []
    store = FeatureHolder(samples)
    for line in stdin:
        if not line or line[0] == "#":  # skip empty lines and comments
            continue

        seqname, coverage_field = line.rstrip().split("\t", 2)[:2]
        seqnames.append(seqname)
        feature_list = []
        for sample_group in coverage_field.split(" "):
            sample_name, coverage = sample_group.split(":", 2)[:2]
            coverage = map(int, coverage.split(","))  # TODO: use sparse numpy objects...
            feature_list.append((sample_name, coverage))
        store.deposit(feature_list)

    # define classes
    #coverages = (2, 3, 4, 5, 11, 20, 46)
    #coverages = (8, 12, 16, 20, 44, 80, 184)
    #sample_number = len(store.samples)
    classes = tuple(PoissonComponent(clist) for clist in coverages)

    # ML-classify
    for i, data in enumerate(izip(*store._sample_columns)):
        log_likelihoods = map(lambda c: c.log_likelihood(data), classes)
        i1, i2 = argmax_two(log_likelihoods)
        likediff = log_likelihoods[i1] - log_likelihoods[i2]
        stdout.write("%s\t%s\t%.2f\n" % (seqnames[i], classes[i1].get_label(), likediff))
