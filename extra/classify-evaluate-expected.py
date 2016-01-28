#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Take two log-transformed (un-normalized) PMF vectors over
a number of classes of whose the first is the label (true) and
the second the predicted distribution. Then, determine the overlap
in terms of expected probability of correct prediction.
Also determine the optimal smoothing parameter alpha for the
predicted array which controls how spiky the distribution is.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import common
import numpy as np
from itertools import count
from sys import argv, stdin, stdout, stderr

stepsize = 1.0

if __name__ == "__main__":
    alpha_max = float(argv[3])

    # load data
    data = []
    for d in (open(argv[1], "r"), open(argv[2], "r")):
        rows = []
        for v in common.parse_lines(d):
            v = np.asarray(v, dtype=common.prob_type)
            rows.append(np.asarray(v, dtype=common.prob_type))
        data.append(-np.vstack(rows))  # stored as -log(P)
    assert(data[0].shape == data[1].shape)


  # calculate probability
    data[0] = common.exp_normalize(data[0])

    optimal_expect = 0.0
    optimal_alpha = None
    alpha = 1.0
    while alpha <= alpha_max:  # TODO: use matrix arithmetic to caculate likelihood for all alphas at once
        stderr.write("alpha: %.2f\n" % alpha)
        tmp = common.exp_normalize(alpha*data[1])
        # np.seterr(divide="ignore")
        joint = np.exp(np.log(data[0]) + np.log(tmp))  # must have same dimension
        stderr.write("average predicted: [%s]\n" % ",".join(["%.2f" % i for i in np.mean(tmp, axis=0)]))
        expect = np.mean(joint.sum(axis=1))
        stderr.write("expected: %.2f\n" % expect)
        if expect > optimal_expect:
            optimal_expect = expect
            optimal_alpha = alpha
        alpha += stepsize

    stdout.write("%.2f\t%.2f\n" % (optimal_alpha, optimal_expect))
