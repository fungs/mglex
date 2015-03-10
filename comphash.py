#!/usr/bin/env python

u"""
Explore a new way to measure nucleotide composition.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"


def usage():
    stderr.write("Usage: %s [--format fasta --normalize --cumulative]\n" % argv[0])


# primitives for hashing

binmap = {
    "a": (1, 0),
    "t": (0, 1),
    "c": (1, 1),
    "g": (0, 0)
}


def nuc2bin(s):
    for c in s:
        try:
            yield binmap[c]
        except KeyError:
            pass  # skip unknown characters


xor = {
    (0, 0): 0,
    (1, 1): 0,
    (1, 0): 1,
    (0, 1): 1
}


togglebits = {
    (0, 0): (1, 1),
    (1, 1): (0, 0),
    (1, 0): (0, 1),
    (0, 1): (1, 0)
}


def revcomp(binmer):
    if len(binmer) == 1:
        yield (0, 0)
    for c in reversed(binmer):
        yield togglebits[c]


# patterns = {
#     1: (((1, 1),),),
#     2: (((1, 0), (1, 0)), ((0, 1), (0, 1))),
#     3: (((1, 0), (1, 0), (1, 0)), ((0, 1), (0, 1), (0, 1)), ((1, 0), (0, 1), (1, 0))),
#     4: (((1, 0), (1, 0), (1, 0), (1, 0)), ((0, 1), (0, 1), (0, 1), (0, 1)),
#         ((0, 1), (1, 0), (1, 0), (0, 1)), ((1, 0), (1, 0), (0, 1), (0, 1))),
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

patterns = dict([(1,(((1, 1),),))] + [(k,(((1,1),)*(k-1)+((1,0),), ((1,1),)*(k-1) + ((0,1),))) for k in range(2,6)])
#patterns = dict([(k,(((1,1),)*(k-1)+((1,0),), ((1,1),)*(k-1) + ((0,1),))) for k in range(1,6)])


def bin2hash(binmer, patterns):
    k = len(binmer)
    try:
        pats = patterns[k]
    except KeyError:
        stderr.write("Could not find patterns for k=%i (kmer is %s)" % (k, binmer))
        exit(1)
    h = [0 for p in pats]  # number of bits
    # print "empty hash", h
    for i, pat in enumerate(pats):
        # print pat
        for j, (pcol, s1col, s2col) in enumerate(zip(pat, revcomp(binmer), binmer)):
            for l in (0, 1):
                if pcol[l]:
                    tmpbit = xor[s1col[l], s2col[l]]
                    h[i] = xor[h[i], tmpbit]
            # print "hash column", h
        # print "hash bit for pattern", h[i]
    # print "complete hash for kmer", k, h
    return tuple(h)


def feature2string(f, norm):
    if norm:
        total = float(sum(f))
        return ",".join(map(lambda x: "%.8f" % (x/total), f))
    return ",".join(map(lambda x: "%i" % x, f))


featurelist2string = lambda flist, norm: "\t".join([feature2string(f, norm) for f in flist])


if __name__ == "__main__":
    from Bio import SeqIO
    import getopt
    from itertools import repeat, product
    from sys import stdin, stderr, stdout, argv

    # parse command line options
    try:
        (opts, args) = getopt.getopt(argv[1:], "hi:nc",
                                     ["help", "informat=", "normalize", "cumulative"])
    except getopt.GetoptError, err:
        print str(err)  # will print something like "option -a not recognized"
        usage()
        exit(2)

    # default parameters
    default_format = "fasta"
    iformat = ""
    valid_formats = {"fasta", "fastq"}
    normalize = False
    cumulative = False
    K = len(patterns)

    for (o, a) in opts:
        if o in ("-h", "--help"):
            usage()
            exit()
        elif o in ("-i", "--informat"):
            iformat = a.lower()
            if iformat not in valid_formats:
                stderr.write("You must specify format as one of: %s\n" % ", ".join(valid_formats))
                usage()
                exit(1)
        elif o in ("-n", "--normalize"):
            normalize = True
        elif o in ("-c", "--cumulative"):
            cumulative = True
        else:
            assert False, "unhandled option"

    if not iformat:
        iformat = default_format

    # handle broken pipes
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    cumulative_features = [[0]*2*len(patterns[k]) for k in range(1, K+1)]  # TODO: numpy arithmetics

    for rec in SeqIO.parse(stdin, iformat):
        # stderr.write("Calculating features for %s\n" % rec.id)
        seq = list(nuc2bin(rec.lower()))
        n = len(seq)

        features = []
        for k in range(1, K+1):
            count = {}
            # stderr.write("k=%i\n" % k)
            # t = 0
            for i in range(n-k):
                h = bin2hash(seq[i:i+k], patterns)
                try:
                    count[h] += 1
                except KeyError:
                    count[h] = 1
                # t += 1

            f = []  # TODO: make numpy array

            for h in product(*repeat((1, 0), len(patterns[k]))):
                # stderr.write("Feature %s\n" % "".join(map(str, h)))
                f.append(count.get(h, 0))

            features.append(f)

        if not cumulative:
            stdout.write("%s\t%s\n" % (rec.id, featurelist2string(features, normalize)))
        else:
            for i, f in enumerate(features):  # TODO: numpy arithmetics
                for j, c in enumerate(f):
                    cumulative_features[i][j] += c

    if cumulative:
        stdout.write("\t%s\n" % featurelist2string(cumulative_features, normalize))

        # stderr.write("normalization constant %i\n" % total)
        # total = float(total)

        # stdout.write("%s\t%.8f,%.8f" % (rec.id, count.get((0,), .0)/total, count.get((1,), .0)/total))
        # stdout.write(rec.id)
        # for k in range(K_lower, K_upper+1):
        #     features = []
        #     stdout.write("\t")
        #     for h in product(*repeat((1, 0), k)):
        #         features.append(count[k].get(h, 0)/total[k])
        #     if normalize:
        #         stdout.write(",".join(map(lambda f: "%.8f" % f, features)))
        #     else:
        #         stdout.write(",".join(map(lambda f: "%i" % f, features)))
        # stdout.write("\n")

        # stdout.write("%s\t%s\t%.8f\n" % (rec.id, "".join(map(str, k)), v/total))
        # for k, v in count.iteritems():
            # if k != (1,):
            #     stdout.write("%s\t%s\t%.8f\n" % (rec.id, "".join(map(str, k)), v/total))
