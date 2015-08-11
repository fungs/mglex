#!/usr/bin/env python3

u"""
 Usage: simplelabel-weightcalc.py [-h] [INPUTFILE]

 -h --help    show this

 Reads from a file or stream which contains the simplelabel format, one entry per line and outputs the same format
 but with different weights for each sub-level of the input path. The weights are inferred from the original weight
 but multiplied with a level factor. The weights are to be used as information for probabilistic modelling.

 Example input:

 1.2.3:100

 Example output:
 1:100,1.2:200,1.2.3:300
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from collections import defaultdict
import sys

def calculate_path_weigths(path, weight):
    partial_path = []
    for i, entry in zip(count(1), path):
        partial_path.append(entry)
        yield partial_path, i*weight  # simple formula putting more weight to the end of the path


print_path_weigth = lambda path, weight: "%s:%i" % (print_path(path), weight)


if __name__ == "__main__":
    import fileinput
    from docopt import docopt

    arguments = docopt(__doc__)

    for line in fileinput.input():
        if line and line[0] == "#":
            continue

        total_per_path = defaultdict(lambda: 0)

        for entry in line.rstrip().split(","):
            path, weight = entry.split(":")
            weight = int(weight)
            for p, w in calculate_path_weigths(path, weight):
                total_per_path[p] += w

        print(total_per_path)
