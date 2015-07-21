#!/usr/bin/env python3

u"""
Takes a label matrix one-zero entries and probability class assignments and calculates an evaluation statistic
S = log((1/C) * \sum_i=1_C (1/|C_i|*(|C_i|-1)) * \sum_{d_1, d_2 \element C_i, d_1 != d_2} p(d_1|C_i)*p(d_2|C_i))
The expected (log-)probability that any two linked contigs (prior knowledge) are grouped together in a cluster.

This measure can be generalized to pairs of sequences which should _not_ belong together in a cluster (between)
and for fuzzy label distributions.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import sys
import numpy as np
from itertools import permutations
from common import print_probmatrix, print_probvector, pretty_probvector


if __name__ == "__main__":
    try:
        filein2 = open(sys.argv[2], "r")
    except IndexError:
        filein2 = sys.stdin

    #joint_clustering_probs = None
    #group_size = None
    predictions_per_labelgroup = {}

    for line1, line2 in zip(open(sys.argv[1], "r"), filein2):

        empty = (not line1, not line2)

        if all(empty):
            continue

        if any(empty):
            sys.stderr.write("Cannot have empty line in one out of two inputs.\n")
            sys.exit(1)

        comment = (line1[0] == "#", line2[0] == "#")

        if all(comment):
            continue

        if any(comment):
            sys.stderr.write("Cannot have comment line in one out of two inputs.\n")
            sys.exit(2)

        fields = line1.rstrip().split("\t")
        try:
            labelvec = np.asarray(fields, dtype=float)
        except ValueError:  # TODO: better handling
            sys.stderr.write("Cannot parse line: %s" % line1)
            sys.exit(1)
 
        assert(sum(labelvec == float("Inf")) == (labelvec.size - 1))
        labelindex = np.nonzero(labelvec == 0.)[0][0]  # only one index can be one by definition
        #assert(labelvec.sum() == 1.)

        fields = line2.rstrip().split("\t")
        predictionvec = -np.asarray(fields, dtype=float)  # logprob row vector

        try:
            #group_size[labelvec] += 1
            predictions_per_labelgroup[labelindex].append(predictionvec)
            #joint_clustering_probs[labelvec] += predictionvec
        except KeyError:
            #group_size = np.zeros(len(labelvec))
            predictions_per_labelgroup[labelindex] = [predictionvec]
            #joint_clustering_probs = np.zeros((len(labelvec), len(predictionvec)))
            #joint_clustering_probs[labelvec] += predictionvec

    # normalize sizes
    #group_size_normalized = (group_size/group_size.sum())[np.newaxis, :]
    # print_probmatrix(group_size_normalized)
    #prob_together_pergroup = np.exp(joint_clustering_probs).sum(axis=1, keepdims=True)
    # print_probmatrix(prob_together_pergroup)
    #expected_prob_overall = np.dot(group_size_normalized, prob_together_pergroup)

    expected_probs_per_group = np.zeros(len(predictions_per_labelgroup))

    for i, rows in enumerate(predictions_per_labelgroup.values()):
        #sys.stdout.write("new label group\n")
        mat = np.vstack(rows)
        group_probs = []
        for v1, v2 in permutations(mat, 2):
             group_probs.append(np.exp(v1+v2).sum())
#            sys.stdout.write("new pair in group\n")
#            print_probvector(v1)
#            print_probvector(v2)
#        print_probmatrix(mat)
#        print_probvector(group_probs)
        mprob = np.mean(group_probs)
        expected_probs_per_group[i] = mprob
#        sys.stdout.write("mean group probability is %.6f\n" % meanprob) 
    mean_prob_overall = np.mean(expected_probs_per_group)
    squared_loss_overall = (expected_probs_per_group**2).sum()
    sys.stdout.write("%.2f\t%.2f\t%s\n" % (mean_prob_overall, squared_loss_overall, pretty_probvector(expected_probs_per_group)))
