#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test classification to Poisson components with coverage.
"""

from numpy import mean

if __name__ == "__main__":
    from sys import argv, exit, stdin, stdout
    from collections import Counter
    from itertools import izip
    
    samples = argv[1:]

    # parse features from stdin
    features_per_sample = []
    for line in stdin:
        if not line or line[0] == "#":  # skip empty lines and comments
            continue

        sample2coverage = {}
        for sample_group in line.rstrip().split(" "):
            sample_name, coverage = sample_group.split(":", 2)[:2]
            coverage = map(int, coverage.split(","))  # TODO: use sparse numpy objects...
            sample2coverage[sample_name] = coverage
        
        stdout.write("%s\n" % (" ".join(["%.2f" % mean(sample2coverage.get(s, .0)) for s in samples])))
