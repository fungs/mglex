#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test classification to Poisson components with coverage.
"""

from numpy import median

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

        seqname, coverage_field = line.rstrip().split("\t", 2)[:2]
        sample2coverage = {}
        for sample_group in coverage_field.split(" "):
            sample_name, coverage = sample_group.split(":", 2)[:2]
            coverage = map(int, coverage.split(","))  # TODO: use sparse numpy objects...
            sample2coverage[sample_name] = coverage
        
        stdout.write("%s\t%s\n" % (seqname, "-".join(["%i" % round(median(sample2coverage.get(s, .0))) for s in samples])))
