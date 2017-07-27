#!/usr/bin/env python3
# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
This script reads a likelihood matrix and applies the given transformation to it.

Usage:
  transform  (--help | --version)
  transform  [--data <file>] [--precision <int>] (--raw-probability|--maximum-likelihood|--posterior|--posterior-ratio|--classindex)

  -h, --help                       Show this screen
  -v, --version                    Show version
  -d <file>, --data <file>         Likelihood matrix; default standard input
  -p <int>, --precision <int>      Output precision; default 2
  -r, --raw-probability            Convert from log to simple representation (small number become zero)
  -m, --maximum-likelihood         Give only the class(es) with the maximum likelihood a non-zero probability
  -p, --posterior                  Normalize the likelihood values over classes (uniform class prior)
  -q, --posterior-ratio            Divide all likelihoods by the maximum likelihood
  -c, --classindex                 Sparsify by reporting the class index of likelihoods above a threshold
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

    # load data input
    if argument["--data"]:  # TODO: block processing
        data = common.load_probmatrix_file(argument["--data"])
    else:
        data = common.load_probmatrix(sys.stdin)

    if argument["--raw-probability"]:
        if data.dtype == types.prob_type:
            np.exp(data, out=data)
        else:
            data = np.exp(data, dtype=types.prob_type)

    if argument["--maximum-likelihood"]:
        maxval = np.nanmax(data, axis=1, keepdims=True)
        mask = data == maxval
        data = -np.inf
        data[np.logical_and(mask, np.isfinite(maxval))] = -np.log(np.sum(mask, axis=1))

    if argument["--posterior"]:
        if data.dtype == types.prob_type:
            common.exp_normalize_inplace(data)
            np.log(data, out=data)
        else:
            tmp = common.exp_normalize(data)
            np.log(tmp, out=data)
    
    if argument["--posterior-ratio"]:
        maxval = np.nanmax(data, keepdims=True)
        data -= maxval
        data[maxval == -np.inf] = -np.inf

    if argument["--class-index"]:
        minval = np.log(1.0)
        for row in data >= minval:
            sys.stdout.write("\t".join(["%i" % i for i in np.where(row)[0]]))
            sys.stdout.write("\n")
        sys.exit(0)

    common.write_probmatrix(data)


if __name__ == "__main__":
    main(sys.argv[1:])
