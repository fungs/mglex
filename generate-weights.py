#!/usr/bin/env python3

u"""
Generate random weights which sum to a total.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import sys
import numpy as np

if __name__ == "__main__":
    weight_num = int(sys.argv[1])
    weight_sum = float(sys.argv[2])
    weight_series = int(sys.argv[3])

    for i in range(weight_series):
        p = np.random.rand(weight_num)
        p /= p.sum()
        weights = weight_sum * p
        sys.stdout.write(" ".join(["%.2f" % w for w in weights]))
        sys.stdout.write("\n")
