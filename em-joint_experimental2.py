#!/usr/bin/env python3
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
    sharpness = int(argv[4])
    seqnames = [line.rstrip() for line in open(argv[1], "r")]
    # print(seqnames)

    seeds = seeds2indices(seqnames, load_seeds(open(argv[2], "r")))
    c = len(seeds)
    # print(seeds)

    # load data
    # stderr.write("parsing coverage features\n")
    # cov_data = coverage.Data(["j"])
    # load_data_file(open(argv[3], "r"), cov_data)

    stderr.write("parsing composition features\n")
    comp_data = composition.Data()
    load_data_file(open(argv[3], "r"), comp_data)

    # data = UniversalData([cov_data, comp_data])
    # data = UniversalData([cov_data])
    data = UniversalData([comp_data])

    # construct inital (hard) responsibilities
    responsibilities = responsibilities_from_seeds(seeds, data.num_data)

    # create a random model
    # model = UniversalModel([coverage.empty_model(c, cov_data.num_features), composition.empty_model(c, comp_data.num_features)])
    # model = UniversalModel([coverage.empty_model(c, cov_data.num_features)])
    model = UniversalModel(sharpness, [composition.empty_model(c, comp_data.num_features)])

    # EM clustering
    priors = flat_priors(model.components)  # uniform (flat) priors
    models, priors, responsibilities = em(model, priors, data, responsibilities)

    # output results if clustering
    stdout.write("#%s\n" % "\t".join(model.names))
    print_predictions(responsibilities)
