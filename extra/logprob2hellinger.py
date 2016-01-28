#!/usr/bin/env python3

u"""
Takes probability class assignments and calculates pairwise Hellinger distances.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from common import print_probmatrix, print_probvector, pretty_probvector


if __name__ == "__main__":
    mat = np.loadtxt(sys.stdin, delimiter="\t")  # read into memory
    mat = np.sqrt(np.exp(-mat))  # transform for use with euclidean distance
    distances = pdist(mat, metric='euclidean')/np.sqrt(2.)

    n = mat.shape[0]

    #weights = -np.log(distances)

    for (i, j), d in zip(combinations(range(1, n+1), 2), distances):
        sys.stdout.write("%i\t%i\t%.2f\n" % (i, j, d))
