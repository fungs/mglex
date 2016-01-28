#!/usr/bin/env python2.7

u"""
Takes a three-column TAB-separated input, where col1=length, col2=label and col3=prediction. Accuracy is output
per sequence length with configurable bin size.
"""

__author__ = "johdro@mpi-inf.mpg.de"

import argparse

if __name__ == "__main__":
    from sys import stdin, stdout, stderr

    bins = {}

    for line in stdin:
        length, label, prediction = line.rstrip().split("\t", 3)[:3]
        length = int(length)

        try:
            bin_count, bin_positions = bins[prediction]
        except KeyError:
            bin_count, bin_positions = bins[prediction] = ([0, 0], [0, 0])

        i = int(label == prediction)
        bin_count[i] += 1
        bin_positions[i] += length

    for pred, (count, pos) in bins.iteritems():
        prec_unweighted = count[1]/float(sum(count))
        prec_weighted = pos[1]/float(sum(pos))
        stdout.write("%s\t%.2f\t%.2f\n" % (pred, prec_unweighted, prec_weighted))
