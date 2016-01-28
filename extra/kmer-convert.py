#!/usr/bin/env python

u"""
Convert output of 'fasta2kmersS -i infile -f outfile -j 4 -k 4 -n 0 -s 0 -h 1'
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from sys import stdin, stdout

if __name__ == "__main__":
    for line in stdin:
        if not line or line[0] == "#":  # skip empty lines and comments
            continue
        fields = line.rstrip().split("\t")
        seqname = fields[-1][fields[-1].find("#")+1:]
        stdout.write("%s\t%s\n" % (seqname, ",".join(fields[:-1])))
