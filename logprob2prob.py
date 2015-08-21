#!/usr/bin/env python3

u"""
Takes as input a vector of log-transformed probabilities and transforms them back for visual inspection.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import sys
import numpy as np
import signal

if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # handle broken pipes

    for line in sys.stdin:
        if not line or line[0] == "#":
            continue

        fields = line.rstrip().split("\t")
        vec = np.asarray(fields, dtype=float)
        sys.stdout.write("\t".join(["%.2f" % i for i in np.exp(-vec)]))
        sys.stdout.write("\n")
