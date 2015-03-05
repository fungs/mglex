#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
# This tiny program should calculate a static index to kmer table.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"


def get_index(kmer):
    h = 0
    for i, j in enumerate(reversed(kmer)):
        h += (4**i)*j
    return h


def kmers(kmax):
    k = 1
    while k <= kmax:
        for kmer_tuple in product("ATGC", repeat=k):
            yield "".join(kmer_tuple)
        k += 1


if __name__ == "__main__":
    from sys import argv, stderr, stdout, exit
    from itertools import product, izip, count
    kmax = int(argv[1])

    stderr.write("calculating index to kmer table for k=%i\n" % kmax)
    stdout.write("\t".join(kmers(int(kmax))))
    stdout.write("\n")
    exit(0)

    for kmer_num, kmer_char in izip(product((0, 1, 2, 3), repeat=k), product('ATGC', repeat=k)):
        index = get_index(kmer_num)
        stdout.write("%s/%s: %i\n" % ("".join(kmer_char), "".join(map(str, kmer_num)), index))
