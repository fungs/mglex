#!/usr/bin/env python3
# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
This is the main program which calculates pairwise bin similarities using mixture likelihoods.

Usage:
  bincompare  (--help | --version)
  bincompare  [--weight <file> --data <file> --responsibility <file> --subset-column <file> --beta <float> --posterior-ratio]

  -h, --help                          Show this screen
  -v, --version                       Show version
  -p, --posterior-ratio               Weigh sequences by full bin posterior; default False
  -d <file>, --data <file>            Likelihood matrix; default standard input
  -r <file>, --responsibility <file>  Responsibility (weight) matrix file; default None
  -w <file>, --weight <file>          Optional weights (sequence length) file; default None
  -s <file, --subset-column <file>    Use subset of column indices (1-based); default None
  -b <float>, --beta <float>          Beta correction factor (e.g. determined via MSE evaluation); default 1.0
  
"""

import sys
import warnings
import numpy as np

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

    # load input
    if argument["--data"]:
        likelihood = common.load_probmatrix_file(argument["--data"])
    else:
        likelihood = common.load_probmatrix(sys.stdin)

    if argument["--beta"]:
        beta = float(argument["--beta"])
        if beta != 1.0:
            with np.errstate(over='ignore'):
                data *= beta

    # load responsibility matrix
    log_responsibility = argument["--responsibility"]
    if log_responsibility is not None:
        log_responsibility = common.load_probmatrix_file(log_responsibility)
    
    # load weights
    log_weight = argument["--weight"]
    if log_weight is not None:
        log_weight = np.log(common.load_seqlens_file(log_weight))
    
    if argument["--posterior-ratio"]:  # scale likelihood proportional to best classification (log(best) = 0)
        with np.errstate(invalid='ignore'):
            likelihood -= np.max(likelihood, axis=1, keepdims=True)
    
    # load subset columns
    subset_cols = argument["--subset-column"]
    if subset_cols is not None:
        with open(subset_cols, "r") as f:
            subset_cols = sorted(set(int(line.rstrip("\n"))-1 for line in f))
            assert subset_cols[0] >= 0
        likelihood = likelihood[:, subset_cols]
        if log_responsibility is not None:
            log_responsibility = log_responsibility[:, subset_cols]

    if not np.all(np.isfinite(np.sum(likelihood, axis=1))):
        warnings.warn("Warning: some sequences have all zero likelihood and are ignored in distance calculations", UserWarning)

    distmat = evaluation.similarity_matrix(likelihood, log_weight=log_weight, log_responsibility=log_responsibility)
    common.write_probmatrix(distmat)

if __name__ == "__main__":
    main(sys.argv[1:])
