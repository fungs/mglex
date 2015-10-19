#!/usr/bin/env python3
# -*- coding: utf-8 -*-

u"""
This is the main program which runs the EM algorithm for compositional and coverage joint clustering.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from common import *
import composition
import binomial
import labeldist
from em import *
from sys import exit, stderr


if __name__ == "__main__":
    from sys import argv, stdin, stdout, stderr
    import signal

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # handle broken pipes

    # parse command line options
    seqnames = load_seqnames(open(argv[1], "r"))
    data_size = load_data_sizes(open(argv[2], "r"))
    seeds = seeds2indices(seqnames, load_seeds(open(argv[3], "r")))
    c = len(seeds)
    weights = [float(i) for i in argv[7:]]

    # load data
    stderr.write("parsing coverage features\n")
    replicate_samples = ("0-0", "1-0") #, "3-0") #, "0-3")
    cov_data = binomial.Data(replicate_samples)
    load_data_file(open(argv[4], "r"), cov_data)

    stderr.write("parsing composition features\n")
    comp_data = composition.Data()
    load_data_file(open(argv[5], "r"), comp_data)

    stderr.write("parsing labeldist features\n")
    lbl_data = labeldist.Data()
    load_data_file(open(argv[6], "r"), lbl_data)

    data = UniversalData([cov_data, comp_data, lbl_data], sizes=data_size)
    assert data.num_features == len(weights)

    # construct inital (hard) responsibilities
    responsibilities = responsibilities_from_seeds(seeds, data.num_data)

    # create empty model for data
    model = UniversalModel(weights, [binomial.empty_model(c, cov_data.num_features),
                                     composition.empty_model(c, comp_data.num_features),
                                     labeldist.empty_model(c, lbl_data.num_features, lbl_data.levelindex)
                                     ])

    # EM clustering
    priors = flat_priors(model.num_components)  # uniform (flat) priors
    models, priors, responsibilities = em(model, priors, data, responsibilities)

    # output results if clustering
    stdout.write("#%s\n" % "\t".join(model.names))
    print_predictions(responsibilities)
