#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
This is the main program which runs the EM algorithm for compositional and coverage joint clustering.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from common import *
import composition
import coverage
from em import *
from sys import exit, stderr


if __name__ == "__main__":
    from sys import argv, stdin, stdout, stderr
    import signal

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # handle broken pipes

    # parse command line options
    seeds = load_seeds(open(argv[1], "r"))
    c = len(seeds)

    # load data
    stderr.write("parsing features\n")
    data = UniversalData([coverage.Data(argv[2:]), composition.Data()])
    data.load(stdin)

    # construct inital (hard) responsibilities
    responsibilities = responsibilities_from_seeds(data, seeds)

    # create a random model
    model = UniversalModel([coverage.random_model(c, data[0].num_features, 1, 50), composition.random_model(c, data[1].num_features)])

    # EM clustering
    priors = flat_priors(model.components)  # uniform (flat) priors
    models, priors, responsibilities = em(model, priors, data, responsibilities)

    # output results if clustering
    print_clusters(responsibilities, data.names, model.names)
