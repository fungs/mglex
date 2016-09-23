#!/usr/bin/env python3

u"""
This is the main program which takes a pre-calculated model file and corresponding sample features. The program returns
the corresponding classification likelihood for each datum and class in the form of a (negatively log-scaled)
likelihood matrix. When normalization is requested, then these are (negatively log-scaled) class posteror
probabilities, also called the responsibility matrix in the context of mixture models.

Usage:
  classify  (--help | --version)
  classify  (--model <file>) [--abcoverage <file>] [--diffcoverage <file>] [--composition <file>]
                             [--labels <file>] [--beta <float>] [--logfile <file>] [--normalize]

  -h, --help                        Show this screen
  -v, --version                     Show version
  -n, --normalize                   Output class posterior instead of the raw likelihood
  -m <file>, --model <file>         Pre-calculated classificaton model file
  -a <file>, --abcoverage <file>    Absolute mean coverage data file for Poisson Model
  -d <file>, --diffcoverage <file>  Differential mean coverage data file for Binomial Model
  -c <file>, --composition <file>   Compositional data (numeric) file for Naive Bayes Model
  -t <file>, --labels <file>        Label-type data file (e.g. a taxonomic path) for Hierarchical Naive Bayes Model
  -b <float>, --beta <float>        Beta correction factor (e.g. determined via MSE evaluation); default 1.0
  -l <file>, --logfile <file>       File for logging
"""

# TODO: support multiple arguments of the same kind, like multiple label input data

import numpy as np
import sys

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


def main(argv):
    from docopt import docopt
    argument = docopt(__doc__, argv=argv, version=__version__)
    common.handle_broken_pipe()

    model = common.load_model_file(argument["--model"])
    data = models.aggregate.AggregateData()

    for arg, submodule, data_opts in (
                ("--abcoverage", models.poisson, {}),
                ("--diffcoverage", models.binomial, {}),
                ("--composition", models.naive_bayes, {}),
                ("--labels", models.hierarchic_naive_bayes, {})):

        filename = argument[arg]

        if filename:
            model_obj = model[len(data)]
            # print(model_obj.context, file=sys.stderr)
            # print(model_obj.labels, file=sys.stderr)
            data_obj = submodule.load_data_file(filename, context=model_obj.context, **data_opts)
            # print(model_obj.context, file=sys.stderr)
            # print(model_obj.labels, file=sys.stderr)
            model_obj.update_context()
            data.append(data_obj)
            # print(data_obj.context, file=sys.stderr)

    if argument["--beta"]:
        model.beta_correction = argument["--beta"]

    mat = model.log_likelihood(data)

    if argument["--normalize"]:
        mat = common.exp_normalize(mat)
        with np.errstate(divide='ignore'):
            common.write_probmatrix(np.log(mat, dtype=types.logprob_type), file=sys.stdout)
    else:
        common.write_probmatrix(mat, file=sys.stdout)


if __name__ == "__main__":
    main(sys.argv[1:])
