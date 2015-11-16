#!/usr/bin/env python3

u"""
 This is the main program which takes a (negatively log-scaled) responsibility (or weight) matrix and corresponding
 sample features. The program trains the composite model by maximum-likelihood fitting and writes the model file to the
 given filename.

Usage:
  train-ml  (--help | --version)
  train-ml  (--seqlen <file>) (--weight <float>) (--outmodel <file>) [--responsibility <file>]
            [--coverage <file>] [--composition <file>] [--labels <file>] [--logfile <file>] [--normalize]

  -h, --help                            Show this screen
  -v, --version                         Show version
  -n, --normalize                       Input weights are normalized (as class posteriors)
  -r <file>, --responsibility <file>    Responsibility (weight) matrix file; default standard input
  -w <float>, --weight <float>          Parameter for logarithmic weighting of sequence length
  -o <file>, --outmodel <file>          Output classificaton model file; default standard input
  -s <file>, --seqlen <file>            Sequence lengths file
  -d <file>, --coverage <file>          Differential mean coverage data file for Binomial Model
  -c <file>, --composition <file>       Compositional data (numeric) file for Naive Bayes Model
  -t <file>, --labels <file>            Label-type data file (e.g. a taxonomic path) for Hierarchical Naive Bayes Model
  -l <file>, --logfile <file>           File for logging
"""

# TODO: support multiple arguments of the same kind, like multiple label input data

__author__ = "johannes.droege@uni-duesseldorf.de"
__version__ = "bla"

import common
import composition
import binomial
import labeldist
import numpy as np
import sys


if __name__ == "__main__":
    from docopt import docopt
    argument = docopt(__doc__, version=__version__)
    common.handle_broken_pipe()

    responsibility_filename = argument["--responsibility"]
    if responsibility_filename:
        responsibility = common.load_probmatrix_file(responsibility_filename)
    else:
        responsibility = common.load_probmatrix(sys.stdin)
    responsibility = np.exp(responsibility, dtype=common.prob_type)

    c = responsibility.shape[1]

    seqlen = common.load_seqlens_file(argument["--seqlen"])
    data = common.UniversalData(sizes=seqlen)
    model = common.UniversalModel(float(argument["--weight"]))

    # TODO: remove data opts by passing seqlen weight vector
    for arg, submodule, data_opts in (
                ("--coverage", binomial, {"sizes": seqlen}),
                ("--composition", composition, {}),
                ("--labels", labeldist, {})):

        filename = argument[arg]
        if filename:
            data_obj = submodule.load_data_file(filename, **data_opts)
            model_obj = submodule.empty_model(c, data_obj)
            data.append(data_obj)
            model.append(model_obj)

    # TODO: assert that the data types fit the models

    model.maximize_likelihood(responsibility, data)
    common.write_model_file(model, argument["--outmodel"])
