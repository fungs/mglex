#!/usr/bin/env python
# enumerate all kmers and calculate hashes for each to determine collision groups,
# then check for equal frequency in higher-level groups

def submers(seq, l):
    for i in range(len(seq) - l):
        yield seq[i:i + l]


from itertools import product, repeat
from comphash import nuc2bin, bin2hash, hash2string, patterns, revcomp
from sys import stdout, exit

# patterns = {
#     1: (((1, 1),),),
#     2: (((1, 0), (1, 0)), ((0, 1), (0, 1))),
#     3: (((0, 1), (1, 0), (0, 1)), ((1, 0), (0, 1), (1, 0))),
#     4: (((0, 1), (1, 0), (1, 0), (0, 1)), ((1, 0), (0, 1), (0, 1), (1, 0))),
# }

# patterns = {
#     1: (((1, 1),),),
#     2: (((1, 0), (1, 0)),
#         ((0, 1), (0, 1))),
#     3: (((0, 1), (1, 0), (0, 1)),
#         ((1, 0), (0, 1), (1, 0))),
#     4: (((0, 1), (1, 0), (1, 0), (0, 1)),
#         ((1, 0), (0, 1), (0, 1), (1, 0)),
#         ((1, 0), (1, 0), (0, 1), (0, 1))),
#     5: (((1, 0), (1, 0), (0, 0), (1, 0), (1, 0)),
#         ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)),
#         ((0, 0), (1, 0), (1, 0), (1, 0), (0, 0)))
# }


#patterns = {
#    1: (((1, 1),),),
#    2: (((1, 1), (1, 0)),
#        ((1, 1), (0, 1))),
#    3: (((1, 1), (1, 1), (1, 0)),
#        ((1, 1), (1, 1), (0, 1))),
#    4: (((1, 1), (1, 1), (1, 1), (1, 0)),
#        ((1, 1), (1, 1), (1, 1), (0, 1))),
#    5: (((1, 1), (1, 1), (1, 1), (1, 1), (1, 0)),
#        ((1, 1), (1, 1), (1, 1), (1, 1), (0, 1)))
#}


K = len(patterns)

# build hashtable
hashtable = [{} for k in range(K)]
for k, kmer2hash in enumerate(hashtable):
    for kmer in product(*repeat("acgt", k+1)):
        kmer_bin = list(nuc2bin(kmer))
        kmer_hash = bin2hash(kmer_bin, patterns)
        kmer_hash_revcomp = bin2hash(revcomp(kmer_bin), patterns)
        if kmer_hash != bin2hash(revcomp(kmer_bin), patterns):
            stdout.write("WARNING: hash %s for k=%i, kmer=%s is different from reverse complement hash %s\n" % (hash2string(kmer_hash), k+1, "".join(kmer), hash2string(kmer_hash_revcomp)))
        kmer2hash[kmer] = kmer_hash
    values = set(kmer2hash.values())
    dim_real = len(values)
    dim_theory = 2**(len(patterns[k+1]))
    stdout.write("There are %i distinct hashes for k=%i: %s\n" % (dim_real, k+1, ",".join(map(hash2string, values))))
    if dim_real != dim_theory:
        stdout.write("WARNING: %i dependent patterns for k=%i\n" % (dim_theory-dim_real, k+1))

# print hashtable
#exit(0)


freqs = [{} for k in range(K)]
# count frequency of smaller kmer hashes in larger kmer hash groups
for k, counter in enumerate(freqs):
    kmer2hash_short = hashtable[k]
    for l in range(k + 1, K):
        kmer2hash_long = hashtable[l]
        counter_l = counter[l] = {}
        for kmer_long, hash_long in kmer2hash_long.iteritems():
            try:
                counter_hash_long = counter_l[hash_long]
            except KeyError:
                counter_hash_long = counter_l[hash_long] = {}
            for kmer_sub in submers(kmer_long, k + 1):
                hash_short = kmer2hash_short[kmer_sub]
                try:
                    counter_hash_long[hash_short] += 1
                except KeyError:
                    counter_hash_long[hash_short] = 1

for k, freqs_khash in enumerate(freqs[:-1]):
    stdout.write("Frequency of each %i-hash\n" % (k + 1))
    for l, freqs_khash_per_ksize in freqs_khash.iteritems():
        stdout.write("...in each %i-hash: " % (l + 1))
        all_values = []
        for binfreqs in freqs_khash_per_ksize.itervalues():
            all_values += binfreqs.values()
            # print binfreqs.values()
        if all(map(lambda v: v == all_values[0], all_values[1:])):
            stdout.write("%i\n" % all_values[0])
        else:
            stdout.write("WARNING: uneven distribution of hashes =>")
            print all_values

#stdout.write("%i\t%s\t%s\n" % (k, "".join(map(str, kmer_hash)), "".join(kmer)))
