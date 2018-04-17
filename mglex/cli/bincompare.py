#!/usr/bin/env python3
# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
This is the main program which calculates pairwise bin similarities using mixture likelihoods.

Usage:
  bincompare  (--help | --version)
  bincompare  [--weight <file> --data <file> --subset-1 <file> --subset-2 <file> --beta <float> --posterior-ratio]
              [--prefilter-thresh <float> --edge-thresh <float>]

  -h, --help                           Show this screen
  -v, --version                        Show version
  -q, --posterior-ratio                Weigh sequences by (subset) bin posterior [default: False]
  -d <file>, --data <file>             Likelihood matrix [standard input]
  -w <file>, --weight <file>           Optional weights (sequence length) file [None]
  -s <file, --subset-1 <file>          Use subset of column indices (1-based) [None]
  -S <file, --subset-2 <file>          Use subset of column indices (1-based) [None]
  -b <float>, --beta <float>           Beta correction factor (e.g. determined via MSE evaluation) [default: 1.0]
  -p <float>, --prefilter-thresh       Contig overlap similarity used to avoid likelihood calculations [default: 0.5]
  -e <float>, --edge-thresh <float>    Only distances <= threshold are reported; use "inf" to show all [default: 30]
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
                likelihood *= beta

    # load weights
    log_weight = argument["--weight"]
    if log_weight is not None:
        log_weight = np.log(common.load_seqlens_file(log_weight))
    
    # load subset columns TODO: refactor
    subset1 = argument["--subset-1"]
    if subset1 is not None:
        with open(subset1, "r") as f:
            subset1 = sorted(set(int(line.rstrip("\n"))-1 for line in f))  # convert to zero-based indexing
            assert subset1[0] >= 0
    subset2 = argument["--subset-2"]
    if subset2 is not None:
        with open(subset2, "r") as f:
            subset2 = sorted(set(int(line.rstrip("\n"))-1 for line in f))  # convert to zero-based indexing
            assert subset2[0] >= 0
    
    # scale likelihood proportional to best classification (log(best) = 0)
    if argument["--posterior-ratio"]:
        with np.errstate(invalid='ignore'):
            likelihood -= np.max(likelihood, axis=1, keepdims=True)
        posterior_scale = True
    else:
        posterior_scale = False

    if not np.all(np.isfinite(np.sum(likelihood, axis=1))):
        warnings.warn("Warning: some sequences have all zero likelihood and are ignored in distance calculations", UserWarning)

    for i, j, dist, ssim in evaluation.binsimilarity_iter(likelihood, log_weight=log_weight, indices=(subset1, subset2),
                                                          prefilter_threshold=float(argument["--prefilter-thresh"]),
                                                          log_p_threshold=-float(argument["--edge-thresh"]),
                                                          posterior_scale=posterior_scale):
        sys.stdout.write("%i\t%i\t%.2f\t%.2f\n" % (i+1, j+1, np.abs(dist), ssim))  # TODO: make precision configurable

if __name__ == "__main__":
    main(sys.argv[1:])
