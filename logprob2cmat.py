#!/usr/bin/env python3

u"""
Takes two distinct probability class assignments and generates a confusion matrix text file for analysis and
visualization.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import sys
import numpy as np
# from itertools import count
from common import print_probmatrix

if __name__ == "__main__":
    cmat = None
    # iteration = count(0)
    title = None

    for line1, line2 in zip(open(sys.argv[1], "r"), open(sys.argv[2], "r")):

        empty = (not line1, not line2)

        if all(empty):
            continue

        if any(empty):
            sys.stderr.write("Cannot have empty line in one out of two inputs.\n")
            sys.exit(1)

        comment = (line1[0] == "#", line2[0] == "#")

        if all(comment):
            continue

        if any(comment):
            sys.stderr.write("Cannot have comment line in one out of two inputs.\n")
            sys.exit(2)

        fields = line1.rstrip().split("\t")
        labelvec = np.exp(-np.asarray(fields, dtype=float))[:, np.newaxis]  # column vector

        fields = line2.rstrip().split("\t")
        predictionvec = np.exp(-np.asarray(fields, dtype=float))[np.newaxis, :]  # row vector

        partial_confusion = np.dot(labelvec, predictionvec)

        try:
            cmat += partial_confusion
        except TypeError:
            cmat = partial_confusion

        # sys.stdout.write("\t".join(["%.2f" % i for i in np.exp(-vec)]))
        # sys.stdout.write("\n")

    # write confusion matrix to output
    try:
        title = sys.argv[3]
    except IndexError:
        title = "generic confusion matrix"

    predictionclasses = [str(i) for i in range(cmat.shape[1])]
    labelclasses = [str(i) for i in range(cmat.shape[0])]

    sys.stdout.write("%s\t" % title)
    sys.stdout.write("\t".join(predictionclasses))
    sys.stdout.write("\n")
    for name, row in zip(labelclasses, cmat):
        sys.stdout.write("%s\t" % name)
        sys.stdout.write("\t".join(["%.2f" % i for i in row]))
        sys.stdout.write("\n")
