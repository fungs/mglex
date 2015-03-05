#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
This is the main program which runs the EM algorithm for compositional clustering.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import composition
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
    data = UniversalData([composition.Data()])
    data.load(stdin)
    single_data = data[0]

    # construct inital (hard) responsibilities
    seed_indices = seeds2indices(seeds, data)
    responsibilities = responsibilities_from_seeds(seed_indices, data.num_data)

    # load initial model
    models = UniversalModel([composition.random_model(c, single_data.num_features)])

    # EM clustering
    priors = flat_priors(models.components)  # uniform (flat) priors
    #models, priors, responsibilities = em(models, priors, data)
    models, priors, responsibilities = em(models, priors, data, responsibilities)

    # plot_clusters_igraph(responsibilities, seed_indices)

    # output results if clustering
    print_responsiblities(responsibilities, data.names, models.names)
