#!/usr/bin/env python3

u"""
 Usage: taxpath2simplelabel.py [-h] [INPUTFILE]

 -h --help    show this

 Reads a two-column file (1. taxon path separated by ",", 2. weight) and converts it to a one-column
 format (1. contig name, 2. short path with weight) which is best suited for re-weighing the label types before
 using the information for probabilistic modelling.

 Example output column:

 1.2.3:100

 where the original data might have been
 taxpath: Baceria;Firmicutes;Bacillales
 weight: 100
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from collections import defaultdict
from itertools import count
import sys

convert_generator_functor = lambda gen: lambda: next(gen)


class LabelIndex:
    def __init__(self):
        self.store = defaultdict(self._context())

    def __getitem__(self, itemseq):
        current = self.store
        path = []
        for item in itemseq:
            index, current = current[item]
            path.append(index)
        return tuple(path)

    def _context(self):
        obj = convert_generator_functor(count())
        return lambda: self._default_value(obj)

    def _default_value(self, obj):
        return obj(), defaultdict(self._context())


print_path = lambda path: ".".join(map(str, path))


def parse_sequence_taxpath_file(inputlines):
    for line in inputlines:
        if line and line[0] == "#":
            continue
        taxpath, weight = line.rstrip().split("\t", 2)[:2]
        if taxpath:
            yield taxpath.split(";"), weight
        else:
            yield (), weight


if __name__ == "__main__":
    import fileinput
    from docopt import docopt

    arguments = docopt(__doc__)

    tree_internal = LabelIndex()

    # convert each line on-the-fly
    for taxpath, weight in parse_sequence_taxpath_file(fileinput.input()):
        if taxpath:
            path = tree_internal[taxpath]
            sys.stdout.write("%s:%s\n" % (print_path(path), weight))
        else:
            sys.stdout.write("\n")
