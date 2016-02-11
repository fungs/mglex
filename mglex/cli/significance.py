#!/usr/bin/env python3

u"""
This script takes two data likelihood matrices with the same number of columns (genomes/classes) and an arbitrary number
of rows (data points). The truth (null hypothesis) matrix is used to derive an empirical log-likelihood distribution
which is weighted via a data weight file (sequence length) and the responsibility matrix of the same shape. Given the
calculated distribution, each likelihood value in the other matrix is converted into a one-sided p-value which is the
rest probability that the value is generated by the null hypothesis model. Doing so, one can exclude data which has a
low probability to reduce the size of the dataset.

Usage:
  classify  (--help | --version)
  classify  (--seqlen <file>) (--nulldata <file>) (--responsibility <file>) [--data <file>]

  -h, --help                           Show this screen
  -v, --version                        Show version
  -s <file>, --seqlen <file>           Sequence lengths file
  -n <file>, --nulldata <file>         Log-likelihood matrix with reference (null hypothesis) values
  -r <file>, --responsibility <file>   Log-likelihood responsibility matrix
  -d <file>, --data <file>             Log-likelihood matrix for which to calculate p-values; default standard input
"""

# TODO: support multiple arguments of the same kind, like multiple label input data

__author__ = "johannes.droege@uni-duesseldorf.de"
__version__ = "bla"

import numpy as np
from itertools import count
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


class PValue(object):
    def __init__(self, ll_sorted, pval_sorted):
        self.ll = ll_sorted
        self.pval = pval_sorted
        self.i = 0

    def get(self, ll):  # make except numpy 1d array
        for i in range(self.i, len(self.ll)):
            if ll >= self.ll[i]:
                self.i = i
                return self.pval[i]
        return 0.0


def main(argv):
    from docopt import docopt
    argument = docopt(__doc__, argv=argv, version=__version__)
    common.handle_broken_pipe()

    # steps
    # 1) load nulldata, responsibility and weights
    # 2) combine responsibility and weights into one matrix
    # 3) compress equal likelihood values and sum up weights
    # 4) compress nulldata columns and calculate corresponding p-value vectors
    # 5) throw away responsibiltiy and weights and original nulldata matrix
    # 6) load data and create sorted view by decreasing likelihood
    # 7) replace likelihood by p-value per columns
    # TODO: use vector arithmetics if possible, reduce memory overhead?

    # load refmatrix and calculate distributions
    nulldata = common.load_probmatrix_file(argument["--nulldata"])  # log-likelihood matrix
    responsibility = common.load_probmatrix_file(argument["--responsibility"])  # log-likelihood matrix
    weights = common.load_seqlens_file(argument["--seqlen"])
    weights = types.prob_type(weights/weights.sum(dtype=types.large_float_type))
    responsibility = types.prob_type(np.exp(responsibility) * weights)

    pvalues = []
    for lcol, rcol in zip(nulldata.T, responsibility.T):
        index_compr = np.where(rcol > 0)
        ll = lcol[index_compr]
        index_compr_sort = ll.argsort()[::-1]
        ll = ll[index_compr_sort]
        rr = rcol[index_compr][index_compr_sort]
        rr /= rr.sum(dtype=types.large_float_type)  # use types.prob_type?

        if ll.size > 0:
            mask = np.ones(ll.size, dtype=np.bool)
            rr[0] = 1.0 - rr[0]
            for i in range(1, ll.size):
                if ll[i] == ll[i-1]:
                    mask[i-1] = False
                rr[i] = max(rr[i-1] - rr[i], 0.0)

            rr = rr[mask]  # remove double entries
            rr[1:] = rr[:-1]
            rr[0] = 1.0
            ll = ll[mask]
            # assert np.all(rr>=0.0)
        pvalues.append(PValue(ll, rr))

    del nulldata, responsibility

    # load data input
    if argument["--data"]:
        data = common.load_probmatrix_file(argument["--data"])
    else:
        data = common.load_probmatrix(sys.stdin)

    for col, pval in zip(data.T, pvalues):
        si = col.argsort()[::-1]
        for i in si:
            col[i] = pval.get(col[i])

    common.write_probmatrix(np.log(data))

if __name__ == "__main__":
    main(sys.argv[1:])