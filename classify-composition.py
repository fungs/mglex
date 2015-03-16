#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
Test classification to frequency components with composition.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import composition,common
import numpy as np
from itertools import izip
from sys import argv, stdin, stdout, stderr


if __name__ == "__main__":
    # parameters
    smoothing_factor = 10 # values > 1 make the distribution more spiky

    # load model
    model = common.UniversalModel(composition.load_model(open(argv[1], "r")))
    
    # model = composition.load_model(open(argv[1], "r"))

    # load data
#    stderr.write("parsing features\n")
    data = common.UniversalData([composition.Data() for m in model])
    # data = composition.Data()

    dnames, data = common.load_data(stdin, data)

    # ML-classify
#    stderr.write("classifying\n")
    log_likelihood = model.log_likelihood(data)

#    print data[0].frequencies
#    print model[0].variables
#    print model[0]._loglikes
#    print model[0].features_used
#    print log_likelihood
#    from numpy import exp
#    print exp(log_likelihood)

    membership = common.exp_normalize(smoothing_factor*log_likelihood)

#    print membership



    # print header line
    stdout.write("#%s\n" % "\t".join(model.names))

    for d, m in izip(dnames, np.asarray(membership)):
#        (i1, L1), (i2, L2) = common.argmax(m, n=2)
#        assert(i1 != i2)
#        stdout.write("%s\t%s\t%s\t%.2f\n" % (d, model.names[i1], model.names[i2], L1-L2))
        stdout.write("%s\t%s\n" % (d, "\t".join(map(lambda x: "%.2f" % x, m))))
