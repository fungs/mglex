#!/usr/bin/env python3

u"""
This is the main program which takes a raw (negatively log-scaled) likelihood matrix and a class soft class assignment
matrix (responsibility) and corresponding weights, e.g. sequence lengths. Each input row corresponds to one datum  and
each column corresponds to a class/genome.

Method "co-clustering":
 Calculates the evaluation statistic
 S = log((1/C) * \sum_i=1_C (1/|C_i|*(|C_i|-1)) * \sum_{d_1, d_2 \element C_i, d_1 != d_2} p(d_1|C_i)*p(d_2|C_i))
 The expected (log-)probability that any two probalistically linked contigs (prior knowledge) are grouped together in a
 cluster.

Method "separation":
 For each class, the likelihood distribution is evaluated and a statistic
 of how well the null hypothesis (positive class) distribution is separated from the alternative hypothesis
 distribution (negative class/other data). The statistic can aid in comparing and selecting appropriate models which
 transform raw data into observation likelihoods. It is the mean-squared error of all classes, where each class error
 is the summation of multiplied error probabilities (p-values of the two distributions) when dividing the data into two
 classes at a specific likelihood value.This measure can be generalized to pairs of sequences which should _not_ belong
 together in a cluster (between) and for fuzzy label distributions.

Method "mse":
 Mean squared error is a fast evaluation measure which is the summed squared difference per datum between the true
 (responsibility) posterior and the predicted posterior distribution. Input likelihood must be normalized so that it
 sums to one over all classes.

Usage:
  classify  (--help | --version)
  classify  (--responsibility <file>) (--method <method>) (--weight <file>)
            [--likelihood <file>] [--subsample <int>] [--random-seed <int>] [--beta <from(:to:step)>]...

  -h, --help                                    Show this screen
  -v, --version                                 Show version
  -l <file>, --likelihood <file>                Likelihood matrix; default standard input
  -r <file>, --responsibility <file>            Responsibility (weight) matrix file
  -w <file>, --weight <file>                    Weights (sequence length) file
  -m <method>, --method <method>                Evaluation method; one of "mse", "co-clustering", "separation"
  -s <int>, --subsample <int>                   Subsample this number of data points for faster calculation
  -z <int>, --random-seed <int>                 Seed for random operations
  -b <from(:to:step)>, --beta <from(:to:step)>  Beta correction factor(s) to evaluate; default 1.0
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

__author__ = "johannes.droege@uni-duesseldorf.de"
from mglex import __version__

methods = {"separation": evaluation.twoclass_separation,
           "co-clustering": evaluation.expected_pairwise_clustering,
           "mse": evaluation.mean_squarred_error
          }


def main(argv):
    from docopt import docopt
    argument = docopt(__doc__, argv=argv, version=__version__)
    common.handle_broken_pipe()

    # print(argument["--beta"])

    subsample = argument["--subsample"]
    try:
        subsample = int(subsample)
    except TypeError:
        pass

    if argument["--beta"]:
        betalist_tmp = set()
        for s in argument["--beta"]:
            if ':' in s:
                fr, to, step = [float(f) for f in s.split(":")]
                betalist_tmp |= frozenset(np.arange(fr, to, step))
            else:
                betalist_tmp.add(float(s))
        betalist = sorted(betalist_tmp)

    else:
        betalist = [1.0]

    # load input
    if argument["--likelihood"]:
        likelihood = common.load_probmatrix_file(argument["--likelihood"])
    else:
        likelihood = common.load_probmatrix(sys.stdin)

    responsibility = common.load_probmatrix_file(argument["--responsibility"])
    weights = common.load_seqlens_file(argument["--weight"])
    # weights = 100.0*np.asarray(weights/weights.max(), dtype=types.prob_type)  # TODO: refactor

    n = likelihood.shape[0]
    if subsample and subsample < n:  # random subsampling, if requested
        try:
            common.set_random_seed(int(argument["--random-seed"]))
        except TypeError:
            sys.stderr.write("No random seed given, consider setting a random seed for better reproducibility.\n")

        rows_sampling = np.random.choice(n, subsample, replace=False)
        likelihood = likelihood[rows_sampling]
        responsibility = responsibility[rows_sampling]
        weights = weights[rows_sampling]

    if argument["--method"] == "separation":  # no exp and no beta scaling
            score = methods[argument["--method"]](likelihood, responsibility, weights)
            sys.stdout.write("%.4f\n" % (score))
    else:
        responsibility = np.exp(responsibility)
        for beta in betalist:
            likelihood_tmp = common.exp_normalize(beta*likelihood)
            score = methods[argument["--method"]](likelihood_tmp, responsibility, weights, logarithmic=False)
            sys.stdout.write("%.2f\t%.6f\n" % (beta, score))


if __name__ == "__main__":
    main(sys.argv[1:])
