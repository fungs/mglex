#!/usr/bin/env python

u"""
Takes to PGM files (ASCII version) of the same dimension and write a PGM file with the difference of both files.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

is_comment = lambda l: l and l[0] == "#"

import numpy as np

if __name__ == "__main__":
    from sys import stdout, stderr, argv, exit
    from itertools import izip

    f1 = open(argv[1], "r")
    f2 = open(argv[2], "r")

    source = izip(f1, f2)
    l1, l2 = source.next()

    # first line match
    if l1 != l2:
        stderr.write("Error in header, first line doesn't match\n")
        exit(1)
    stdout.write(l1)

    # eat comment lines
    l1, l2 = source.next()
    while is_comment(l1):
        l1 = f1.next()
    while is_comment(l2):
        l2 = f2.next()
    stdout.write("# CREATOR pgmdiff.py\n")

    # dimension line match
    if l1 != l2:
        stderr.write("Error in header, dimensions are different\n")
        stderr.write(l1)
        stderr.write(l2)
        exit(1)
    stdout.write(l1)

    for l1, l2 in source:
        v1 = np.array(l1.rstrip().split(" "), dtype=np.int8)
        v2 = np.array(l2.rstrip().split(" "), dtype=np.int8)
        diff = np.abs(v1 - v2)
        # print diff
        stdout.write(" ".join(map(str, diff)))
        stdout.write("\n")
