#!/usr/bin/env python3

u"""
This is the main program which takes single space separated a list of sequence identifiers per line which define the
classes or clusters. The program returns a global (negatively log-scaled) responsibility matrix.
.

Usage:
  build_responsibility (--help | --version)
  build_responsibility  (--seeds <file>) [--identifiers <file>]

  -h, --help                        Show this screen
  -v, --version                     Show version
  -s <file>, --seeds <file>         Space-separated sequence identifier per line
  -i <file>, --identifiers <file>   Sequence identifiers, one per line. Defaults to standard input.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"
__version__ = "bla"

import common
import sys

if __name__ == "__main__":
    from docopt import docopt
    argument = docopt(__doc__, version=__version__)

    seeds = common.load_seeds_file(argument["--seeds"])

    seqnames_filename = argument["--identifiers"]
    if seqnames_filename:
        seqnames = common.load_seqnames_file(filename)
    else:
        seqnames = common.load_seqnames(sys.stdin)

    responsibility = common.seeds2responsibility_iter(seqnames, seeds)
    common.write_probmatrix_iter(responsibility, sys.stdout)
