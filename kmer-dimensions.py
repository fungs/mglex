#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
# This tiny program to investigate into the number and shape of the independent kmer space.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"


revcomptable = {"A": "T",
                "T": "A",
                "G": "C",
                "C": "G"}


def get_index(kmer):
    h = 0
    for i, j in enumerate(reversed(kmer)):
        h += (4**i)*j
    return h


def kmers(kmax, kmin=None):
    if not kmin:
        k = kmax
    else:
        k = kmin
    while k <= kmax:
        for kmer_tuple in product("ATGC", repeat=k):
            yield "".join(kmer_tuple)
        k += 1


def reverse_complement(kmer):
    return "".join(map(lambda c: revcomptable[c], reversed(kmer)))


if __name__ == "__main__":
    from sys import argv, stderr, stdout, exit
    from itertools import product, izip, count
    k = int(argv[1])
    seen = set()
    pallindromes = 0
    total = 0

    for kmer in kmers(k):
        rckmer = reverse_complement(kmer)
        if rckmer not in seen:
            seen.add(kmer)
        if rckmer == kmer:
            pallindromes += 1
        total += 1
    # print seen
    stdout.write("k: %i, # kmers: %i, # without rcomp.: %i, # pallindromes: %i, hypthetical number of ind. dimensions: %.2f\n" % (k, total, len(seen), pallindromes, len(seen)-pallindromes/2))
