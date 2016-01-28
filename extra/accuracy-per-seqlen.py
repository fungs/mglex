#!/usr/bin/env python2.7

u"""
Takes a three-column TAB-separated input, where col1=length, col2=label and col3=prediction. Accuracy is output
per sequence length with configurable bin size.
"""

__author__ = "johdro@mpi-inf.mpg.de"

import argparse

if __name__ == "__main__":
    from sys import stdin, stdout, stderr

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--binsize", metavar='N', type=int,
                        required=True, help="length bins for accuracy calculation")
    args = parser.parse_args()

    positions_correct = 0
    positions_incorrect = 0
    binborder = args.binsize
    bin_correct = 0
    bin_incorrect = 0
    for line in stdin:
        length, label, prediction = line.rstrip().split("\t", 3)[:3]
        length = int(length)

        # count bin accuracy
        if length > binborder:
            if bin_correct or bin_incorrect:
                stdout.write("%i\t%i\t%i\n" % (binborder, bin_correct, bin_correct+bin_incorrect))
            bin_correct = bin_incorrect = 0
            binborder = length - length % args.binsize + args.binsize

        # count overall positional accuracy
        if label == prediction:
            bin_correct += 1
            positions_correct += length
        else:
            bin_incorrect += 1
            positions_incorrect += length

    stderr.write("total weighted classification accuracy is %.2f\n" % (positions_correct/float(positions_correct+positions_incorrect)))
