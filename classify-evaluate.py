#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
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
from sys import argv, stdin, stdout, stderr


if __name__ == "__main__":
    # parameters
    alpha = 1.0

    # load data
    data = []
    for d in (open(argv[1], "r"), open(argv[2], "r")):
        rows = []
	for v in common.parse_lines(d):
            v = np.asarray(vec, dtype=common.prob_type)
            rows.append(np.asarray(vec, dtype=common.prob_type))
        data.append(np.vstack(rows))
    assert(data[0].shape == data[1].shape)

    # determine optimal smoothing parameter
    # TODO
    
    # calculate probability
    data[1] = np.log(common.exp_normalize(data[1]))  # normalize vectors
    joint = data[0] + data[1]  # must have same dimension
    joint = exp(joint)
    joint = joint.sum(axis=1)  # TODO: check axis
    joint = log(joint).sum()  # check sum of all points or product?
    stdout.write("Joint probability: %f\n" % joint)
