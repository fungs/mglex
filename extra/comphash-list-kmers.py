#!/usr/bin/env python
# enumerate all kmers and calculate hashes for each to determine collision groups

from itertools import product, repeat
from comphash import nuc2bin, bin2hash, patterns
from sys import stdout

for k in range(1, len(patterns)+1):
    for kmer in product(*repeat("acgt", k)):
        kmer_bin = list(nuc2bin(kmer))
        kmer_hash = bin2hash(kmer_bin, patterns)
        stdout.write("%i\t%s\t%s\n" % (k, "".join(map(str, kmer_hash)), "".join(kmer)))
