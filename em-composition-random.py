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
    c = int(argv[1])

    # load data
    stderr.write("parsing features\n")
    data = UniversalData([composition.Data()])
    data.load(stdin)
    single_data = data[0]

    # load initial model
    #model = load_model(open(argv[1], "r"))
    models = UniversalModel([composition.random_model(c, single_data.num_features)])

    # EM clustering
    priors = flat_priors(models.components)  # uniform (flat) priors
    models, priors, responsibilities = em(models, priors, data)

    # output results if clustering
    print_clusters(responsibilities, data.names, models.names)
