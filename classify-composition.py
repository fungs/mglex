#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
Test classification to frequency components with composition.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"


if __name__ == "__main__":
    from composition import *
    from itertools import izip
    from sys import argv, stdin, stdout, stderr

    # load model
    model = load_model(open(argv[1], "r"))

    # load data
    stderr.write("parsing features\n")
    data = load_data(stdin)

    # ML-classify
    stderr.write("classifying\n")
    log_likelihoods = model.log_likelihood(data)

    for name, membership in izip(data.seqnames, np.asarray(log_likelihoods)):
        (i1, L1), (i2, L2) = argmax(membership, n=2)
        assert(i1 != i2)
        stdout.write("%s\t%s\t%s\t%.2f\n" % (name, model.names[i1], model.names[i2], L1-L2))