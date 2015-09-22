#!/usr/bin/env python3

u"""
Takes as input a vector of log-transformed probabilities and transforms it to maximum likelihood (sharp) predictions.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import sys
import numpy as np
import signal
#from itertools import chain
from common import argmax

if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # handle broken pipes

    min_log = float("-Inf")

    try:
        min_log = np.log(float(sys.argv[1]))
    except:
        pass

    for line in sys.stdin:
        if not line or line[0] == "#":
            continue

        fields = line.rstrip().split("\t")
        vec_log = -np.asarray(fields, dtype=float)
        max_log = vec_log.max()
        if max_log >= min_log:
            mask = vec_log == max_log
            vec_log[mask] = -np.log(mask.sum())
            vec_log[np.logical_not(mask)] = float("-Inf")
        sys.stdout.write("\t".join(["%.2f" % i for i in -vec_log]))
        sys.stdout.write("\n")
