#!/usr/bin/env python

u"""
Converts a FASTA file to an bit image in PBM format.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import numpy as np


def dna2bitvector(dna):
    bv = np.zeros(len(dna), dtype=np.bool_)
    for i, c in enumerate(dna):
        if c in {"G", "C"}:
            bv[i] = 1
    return bv

bool2bitchar = lambda b: str(int(b))


if __name__ == "__main__":
    from Bio import SeqIO
    from sys import stdin, stdout, argv
    from itertools import imap

    data = []
    records = SeqIO.parse(stdin, "fasta")

    for rec in records:
        data.append(dna2bitvector(rec.seq))

    if len(data):
        data = np.concatenate(data)

        if len(argv) > 1 and argv[1] == "mirror":
            out = imap(bool2bitchar, reversed(data))
        else:
            out = imap(bool2bitchar, data)

        stdout.write("P1\n# CREATOR: dna2pbm.py\n")
        stdout.write("%i 1\n" % data.size)
        # stdout.write(map(int, data))
        stdout.write("".join(out))
        stdout.write("\n")
