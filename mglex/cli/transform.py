#!/usr/bin/env python3
# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
This script reads a likelihood matrix and applies the given transformation to it.

Usage:
  transform  (--help | --version)
  transform  [--data <file>] [--beta <float> --precision <int>] [
  --logarithm|--maximum-likelihood|--posterior|--posterior-ratio|--class-index <float>|--raw-probability]

  -h, --help                         Show this screen
  -v, --version                      Show version
  -d <file>, --data <file>           Likelihood matrix; default standard input
  -i <int>, --precision <int>        Output precision; default 2
  -b <float>, --beta <float>         Beta correction factor (e.g. determined via MSE evaluation); default 1.0
  -r, --raw-probability              Show output in normal representation (small number become zero)
  -m, --maximum-likelihood           Give only the class(es) with the maximum likelihood a non-zero probability
  -p, --posterior                    Normalize the likelihood values over classes (uniform class prior)
  -q, --posterior-ratio              Divide all likelihoods by the maximum likelihood
  -l, --logarithm                    Convert from simple to logarithmic format
  -c <float>, --class-index <float>  Report only class indices (one-based) with likelihoods above threshold; default 1.0
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
    # read, process and write input in blocks using block size
    # this will be faster and work better with pipes

    # load data input
    if argument["--data"]:
        data = common.load_probmatrix_file(argument["--data"])
    else:
        data = common.load_probmatrix(sys.stdin)
    
    if argument["--beta"]:
        beta = float(argument["--beta"])
        if beta != 1.0:
            data *= beta

    if argument["--maximum-likelihood"]:  # TODO: speed up
        maxval = np.nanmax(data, axis=1, keepdims=True)
        mask = data == maxval
        data[:] = -np.inf
        mask &= np.isfinite(maxval)
        for drow, mrow, val in zip(data, mask, -np.log(np.sum(mask, axis=1))):
            drow[mrow] = val

    elif argument["--posterior"]:
        if data.dtype == types.prob_type:
            common.exp_normalize_inplace(data)
            np.log(data, out=data)
        else:
            tmp = common.exp_normalize(data)
            np.log(tmp, out=data)
    
    elif argument["--posterior-ratio"]:
        maxval = np.nanmax(data, axis=1, keepdims=True)
        data -= maxval
        data[np.isinf(maxval)[:, 0]] = -np.inf

    elif argument["--class-index"]:
        minval = np.log(float(argument["--class-index"]))
        for row in data >= minval:
            sys.stdout.write(" ".join(["%i" % i for i in np.where(row)[0]]))
            sys.stdout.write("\n")
        sys.exit(0)  # do not output original matrix
    
    if argument["--logarithm"]:  # adjust reading function
        if data.dtype == types.logprob_type:
            np.log(-data, out=data)
        else:
            data = np.log(-data, dtype=types.logprob_type)
    
    if argument["--raw-probability"]:
        if data.dtype == types.prob_type:
            np.exp(data, out=data)
        else:
            data = np.exp(data, dtype=types.prob_type)

    common.write_probmatrix(data)

if __name__ == "__main__":
    main(sys.argv[1:])
