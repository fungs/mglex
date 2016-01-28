#!/usr/bin/env python3

u"""
 Usage: simplelabel-weightcalc.py [-hs] [INPUTFILE]

 -h --help    show this
 -s --sorted  write hierarchical labels in sorted order

 Reads from a file or stream which contains the simplelabel format, one entry per line and outputs the same format
 but with different weights for each sub-level of the input path. The weights are inferred from the original weight
 but multiplied with a level factor. The weights are to be used as information for probabilistic modelling. By
 externalizing the creation of the weights along the taxonomy tree one gains the freedom to set these according to a
 custom scheme, or even by use of phylogenetic (tree) distances.

 Example input:

 1.2.3:100

 Example output:
 1:100 1.2:200 1.2.3:300
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from collections import defaultdict
from itertools import count
import sys


def calc_path_weigths_linear(path, weight):
    u"""
    Calculate linear increasing weights for each level in the label hierarchy, put twice the weight on the second
    level, etc.
    """

    partial_path = tuple()
    for i, entry in zip(count(1), path.split(".")):
        partial_path = partial_path + (entry,)
        yield partial_path, i*weight  # simple formula putting more weight to the end of the path


path_string = lambda path: ".".join(path)
path_weigth_string = lambda path, weight: "%s:%i" % (path_string(path), weight)
simpleweight_string_sorted = lambda path, weight: " ".join(path_weigth_string(p, w)
                                                             for p, w in sorted(total_per_path.items()))
simpleweight_string_unsorted = lambda path, weight: " ".join(path_weigth_string(p, w)
                                                           for p, w in total_per_path.items())


if __name__ == "__main__":
    import fileinput
    from docopt import docopt

    arguments = docopt(__doc__)

    # set sorted output
    if arguments["--sorted"]:
        simpleweight_string = simpleweight_string_sorted
    else:
        simpleweight_string = simpleweight_string_unsorted

    # set input stream
    infile = arguments["INPUTFILE"]
    if infile is None:
        infile = "-"

    # iterate over input
    for line in fileinput.input(infile):
        line = line.rstrip()

        # filter comments
        if line and line[0] == "#":
            continue

        # for non-empty labels
        if line:
            total_per_path = defaultdict(lambda: 0)
            for entry in line.split(" "):
                path, weight = entry.split(":", 2)[:2]
                weight = int(weight)
                for p, w in calc_path_weigths_linear(path, weight):
                    total_per_path[p] += w
            sys.stdout.write(simpleweight_string(p, w))
        sys.stdout.write("\n")
