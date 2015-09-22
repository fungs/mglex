#!/usr/bin/env python3
# -*- coding: utf-8 -*-

u"""
This is the main program which runs the EM algorithm for coverage clustering.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from common import *
import labeldist
from em import *
from sys import exit, stderr


if __name__ == "__main__":
    from sys import argv, stdin, stdout, stderr
    import signal

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # handle broken pipes

    # parse command line options
    weight = float(argv[5])
    seqnames = load_seqnames(open(argv[1], "r"))

    # load data sizes (contig length)
    data_size = load_data_sizes(open(argv[2], "r"))

    # load seeds
    seeds = seeds2indices(seqnames, load_seeds(open(argv[3], "r")))
    c = len(seeds)

    # load data
    stderr.write("parsing labeldist features\n")
    ld_data = labeldist.Data()
    load_data_file(open(argv[4], "r"), ld_data)
    data = UniversalData([ld_data], sizes=data_size)

    # construct inital (hard) responsibilities
    responsibilities = responsibilities_from_seeds(seeds, data.num_data)

    # create a random model
    model = UniversalModel([weight], [labeldist.empty_model(c, ld_data.num_features, ld_data.levelindex)])

    # EM clustering
    priors = flat_priors(model.num_components)  # uniform (flat) priors
    models, priors, responsibilities = em(model, priors, data, responsibilities)

    # output results if clustering
    stdout.write("#%s\n" % "\t".join(model.names))
    print_predictions(responsibilities)
