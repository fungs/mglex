#!/usr/bin/env python3
# This file is subject to the terms and conditions of the GPLv3 (see file 'LICENSE' as part of this source code package)

u"""
This is the main program which takes single space separated a list of sequence identifiers per line which define the
classes or clusters. The program returns a global (negatively log-scaled) responsibility matrix.

Usage:
  buildmatrix  (--help | --version)
  buildmatrix  (--seeds <file>) [--identifiers <file>]

  -h, --help                        Show this screen
  -v, --version                     Show version
  -s <file>, --seeds <file>         Space-separated sequence identifier per line
  -i <file>, --identifiers <file>   Sequence identifiers; one per line; default standard input
"""

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

__author__ = "code@fungs.de"
from mglex import __version__


def main(argv):
    from docopt import docopt
    argument = docopt(__doc__, argv=argv, version=__version__)
    common.handle_broken_pipe()

    seeds = common.load_seeds_file(argument["--seeds"])

    seqnames_filename = argument["--identifiers"]
    if seqnames_filename:
        seqnames = common.load_seqnames_file(seqnames_filename)
    else:
        seqnames = common.load_seqnames_iter(sys.stdin)

    responsibility = common.seeds2responsibility_iter(seqnames, seeds)
    common.write_probmatrix_iter(responsibility, sys.stdout)


if __name__ == "__main__":
    main(sys.argv[1:])
