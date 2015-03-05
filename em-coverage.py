#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
This is the main program which runs the EM algorithm for coverage clustering.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import coverage
from em import *


if __name__ == "__main__":
    from sys import argv, stdin, stdout, stderr
    import signal

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # handle broken pipes

    # parse command line options
    seeds = load_seeds(open(argv[1], "r"))
    c = len(seeds)

    # load data
    stderr.write("parsing features\n")
    data = UniversalData([coverage.Data(argv[2:])])
    data.load(stdin)
    single_data = data[0]

    # construct inital (hard) responsibilities
    responsibilities = responsibilities_from_seeds(data, seeds)

    # load initial model
    models = UniversalModel([coverage.random_model(c, single_data.num_features, 1, 50)])

    # EM clustering
    priors = flat_priors(models.components)  # uniform (flat) priors
    models, priors, responsibilities = em(models, priors, data, responsibilities)

    # output results if clustering
    print_clusters(responsibilities, data.names, models.names)
