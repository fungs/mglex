#!/usr/bin/env python3
# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
This script spreads numeric input sequence features (columns) over classes as defined by a (probability) matrix.

Usage:
  spread  (--help | --version)
  spread  (--responsibility <file>) [--data <file> --normalize --weight <file>]

  -h, --help                          Show this screen
  -v, --version                       Show version
  -d <file>, --data <file>            Feature matrix; default standard input
  -r <file>, --responsibility <file>  Responsibility (weight) matrix file
  -w <file>, --weight <file>          Weights (sequence length) file for normalization
  -i <int>, --precision <int>         Output precision; default 2
  -n, --normalize                     Report (weighted) average instead of sum
"""

import numpy as np
import sys

# some ugly code which makes this run as a standalone script
try:  # when run inside module
    from .. import *
except SystemError:  # when run independently, needs mglex package in path
    try:
        from mglex import *
    except ImportError:
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from mglex import *

__author__ = "code@fungs.de"
from mglex import __version__


def main(argv):
    from docopt import docopt
    argument = docopt(__doc__, argv=argv, version=__version__)
    common.handle_broken_pipe()

    # TODO:
    # process input per column? different data types?

    # load data input
    if argument["--data"]:  # TODO: define type different from probabilities
        data = common.load_matrix_file(argument["--data"], dtype=types.large_float_type)
    else:
        data = common.load_matrix(sys.stdin, dtype=types.large_float_type)
    
    responsibility = common.load_probmatrix_file(argument["--responsibility"])
    np.exp(responsibility, dtype=types.prob_type, out=responsibility)  # TODO: work in log-space?

    scalefactor = 1.

    weight = argument["--weight"]
    if weight:
        seqlen = common.load_seqlens_file(argument["--weight"])
        scalefactor = np.max(seqlen)
        weight = np.asarray(seqlen/scalefactor, dtype=types.prob_type)  # avoid large numbers in prob_type
        np.multiply(responsibility, weight, out=responsibility)

    normalize = argument["--normalize"]
    if normalize:
        np.divide(responsibility, np.sum(responsibility, axis=0, keepdims=True), out=responsibility)  # re-normalize
    
    # features: NxF, probs: NxC -> output: CxF
    spread = np.dot(responsibility.T, data)
    
    if not normalize and scalefactor != 1.:
        np.multiply(spread, scalefactor, out=spread)
        
    common.write_matrix(spread)

if __name__ == "__main__":
    main(sys.argv[1:])
