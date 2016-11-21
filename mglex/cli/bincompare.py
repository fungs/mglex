#!/usr/bin/env python3
# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
This is the main program which calculates pairwise bin similarities using mixture likelihoods.

Usage:
  bincompare  (--help | --version)
  bincompare  (--weight <file>) [--likelihood <file>] [--beta <float>]

  -h, --help                        Show this screen
  -v, --version                     Show version
  -l <file>, --likelihood <file>    Likelihood matrix; default standard input
  -w <file>, --weight <file>        Weights (sequence length) file
  -b <float>, --beta <float>        Beta correction factor (e.g. determined via MSE evaluation); default 1.0
"""

import sys
import numpy as np

# some ugly code which makes this run as a standalone script
try:  # when run inside module
    from .. import *
except SystemError:  # when run independenly, needs mglex package in path
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
    if argument["--likelihood"]:
        likelihood = common.load_probmatrix_file(argument["--likelihood"])
    else:
        likelihood = common.load_probmatrix(sys.stdin)

    if argument["--beta"]:
        likelihood *= types.logprob_type(argument["--beta"])

    weights = common.load_seqlens_file(argument["--weight"])
    distmat = evaluation.similarity_matrix(likelihood, weights)
    common.write_probmatrix(distmat)

if __name__ == "__main__":
    main(sys.argv[1:])
