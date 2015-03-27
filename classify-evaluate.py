#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Take two log-transformed (un-normalized) membership vectors over
a number of classes of whose the first is the label (true) and
the second the predicted distribution. Then, determine the overlap
in terms of probability of correct prediction. Optionally, determine
the optimal smoothing parameter alpha for the predicted array
which controls how spiky the distribution is.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import common
import numpy as np
from itertools import count
from sys import argv, stdin, stdout, stderr


if __name__ == "__main__":

    # load data
    data = []
    for d in (open(argv[1], "r"), open(argv[2], "r")):
        rows = []
        for v in common.parse_lines(d):
            v = np.asarray(v, dtype=common.prob_type)
            rows.append(np.asarray(v, dtype=common.prob_type))
        data.append(-np.vstack(rows))  # stored as -log(P)
    assert(data[0].shape == data[1].shape)

    # determine optimal smoothing parameter
    # TODO
    
    # calculate probability
    data[0] = common.exp_normalize(data[0])

    optimal_likelihood = float("-inf")
    optimal_alpha = None
    for alpha in count(1, 1.):
        tmp = common.exp_normalize(alpha*data[1])
        np.seterr(divide="ignore")
        joint = np.exp(np.log(data[0]) + np.log(tmp))  # must have same dimension
        joint = joint.sum(axis=1)  # TODO: check axis
        likelihood = np.log(joint).sum()  # sum or product of likelihoods?
        stdout.write("%.2f\t%.2f\n" % (alpha, likelihood))
        if likelihood > optimal_likelihood:
            optimal_likelihood = likelihood
            optimal_alpha = alpha
        else:
            break
    stdout.write("optimal=%.2f\t%.2f\n" % (optimal_alpha, optimal_likelihood))
