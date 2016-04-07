#!/usr/bin/env python3

u"""
This is the main program which takes a (negatively log-scaled) responsibility (or weight) matrix and corresponding
sample features. The program trains the composite model by maximum-likelihood fitting and writes the model file to the
given filename.

Usage:
  train  (--help | --version)
  train  (--seqlen <file>) (--outmodel <file>) [--responsibility <file>]
            [--coverage <file>] [--composition <file>] [--labels <file>] [--logfile <file>]

  -h, --help                            Show this screen
  -v, --version                         Show version
  -r <file>, --responsibility <file>    Responsibility (weight) matrix file; default standard input
  -o <file>, --outmodel <file>          Output classificaton model file; default standard output
  -s <file>, --seqlen <file>            Sequence lengths file
  -d <file>, --coverage <file>          Differential mean coverage data file for Binomial Model
  -c <file>, --composition <file>       Compositional data (numeric) file for Naive Bayes Model
  -t <file>, --labels <file>            Label-type data file (e.g. a taxonomic path) for Hierarchical Naive Bayes Model
  -l <file>, --logfile <file>           File for logging
"""

# TODO: support multiple arguments of the same kind, like multiple label input data

__author__ = "johannes.droege@uni-duesseldorf.de"
__version__ = "bla"

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


def main(argv):
    from docopt import docopt
    argument = docopt(__doc__, argv=argv, version=__version__)
    common.handle_broken_pipe()

    responsibility_filename = argument["--responsibility"]
    if responsibility_filename:
        responsibility = common.load_probmatrix_file(responsibility_filename)
    else:
        responsibility = common.load_probmatrix(sys.stdin)
    responsibility = np.exp(responsibility, dtype=types.prob_type)

    n, c = responsibility.shape

    seqlen = common.load_seqlens_file(argument["--seqlen"])
    data = models.aggregate.AggregateData()
    model = models.aggregate.AggregateModel()

    for arg, submodule, data_opts in (
                ("--coverage", models.binomial, {}),
                ("--composition", models.naive_bayes, {}),
                ("--labels", models.hierarchic_naive_bayes, {})):

        filename = argument[arg]
        if filename:
            data_obj = submodule.load_data_file(filename, **data_opts)
            model_obj = submodule.empty_model(c, context=data_obj.context)
            assert data_obj.num_data == n  # TODO: put simple checks into model functions or context
            data.append(data_obj)
            model.append(model_obj)
            # print(data_obj.context, file=sys.stderr)
            # print(model_obj.context, file=sys.stderr)

    weights = np.asarray(seqlen/seqlen.sum(), dtype=types.prob_type)
    model.maximize_likelihood(data, responsibility, weights)
    common.write_model_file(model, argument["--outmodel"])


if __name__ == "__main__":
    main(sys.argv[1:])
