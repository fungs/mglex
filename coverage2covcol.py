#!/usr/bin/env python3

u"""
Read coverage file format and output samples as columns and positions as rows. There will be one file per input row and
the output files are named after the sequence identifier.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import sys
from itertools import repeat

def unpack(e):
    tmp = e.split(":", 2)
    return tmp[0], tmp[1].split(",")

zeros = repeat("0")

if __name__ == "__main__":
    samples = sys.argv[1:]

    for line in sys.stdin:
        if line and line[0] == "#":
            continue

        ident, rest = line.rstrip().split("\t", 2)

        with open("%s.covcol" % ident, "w") as f:
            data = dict((unpack(e) for e in rest.split(" ")))
            for rv in zip(*[data.get(s, zeros) for s in samples]):
                f.write("\t".join(rv))
                f.write("\n")
