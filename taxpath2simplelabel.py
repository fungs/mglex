#!/usr/bin/env python3

u"""
 Usage: taxpath2simplelabel.py [-h] [INPUTFILE]

 -h --help    show this

 Reads a two-column file (1. taxon path separated by ";", 2. weight) and converts it to a one-column
 format (short path with weight) which is best suited for re-weighing the label types before
 using the information for probabilistic modelling.

 Example output column:

 1.2.3:100

 where the original data might have been
  taxpath: Baceria;Firmicutes;Bacillales
  weight: 100
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from common import InternalTreeIndex
import sys

print_path = lambda path: ".".join(map(str, path))


def parse_sequence_taxpath_file(inputlines):
    for line in inputlines:
        if line and line[0] == "#":
            continue
        taxpathstr, weight = line.rstrip().split("\t", 2)[:2]
        if taxpathstr:
            taxpath = taxpathstr.split(";")
            assert all(taxpath)  # do not allow empty fields! This introduces ambiguity but labels must be unique
            yield taxpath, weight
        else:
            yield (), weight


if __name__ == "__main__":
    import fileinput
    import signal
    from docopt import docopt

    arguments = docopt(__doc__)

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # handle broken pipes

    # convert each line on-the-fly
    tree = InternalTreeIndex()
    for taxpath, weight in parse_sequence_taxpath_file(fileinput.input()):
        if taxpath:
            path = tree[taxpath]
            sys.stdout.write("%s:%s\n" % (print_path(path), weight))
        else:
            sys.stdout.write("\n")
