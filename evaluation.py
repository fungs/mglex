u"""
Submodule with all evaluation-related code.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import common
from itertools import permutations
import numpy as np
import sys

u"""
Takes a label matrix one-zero entries and probability class assignments and calculates an evaluation statistic
S = log((1/C) * \sum_i=1_C (1/|C_i|*(|C_i|-1)) * \sum_{d_1, d_2 \element C_i, d_1 != d_2} p(d_1|C_i)*p(d_2|C_i))
The expected (log-)probability that any two linked contigs (prior knowledge) are grouped together in a cluster.

This measure can be generalized to pairs of sequences which should _not_ belong together in a cluster (between)
and for fuzzy label distributions.
"""
def expected_pairwise_clustering(lmat, pmat):
    predictions_per_labelgroup = {}

    for lvec, pvec in zip(lmat, pmat):
        number_nonzero = lvec.size - np.sum(lvec == float("Inf"))
        if number_nonzero == 0:
            continue
        assert number_nonzero == 1

        i = np.nonzero(lvec == 0.)[0][0]  # only one index can be one by definition

        try:
            predictions_per_labelgroup[i].append(pvec)
        except KeyError:
            predictions_per_labelgroup[i] = [pvec]

    expected_probs_per_group = np.zeros(len(predictions_per_labelgroup))

    for i, rows in enumerate(predictions_per_labelgroup.values()):
        mat = np.vstack(rows)
        group_probs = []
        for v1, v2 in permutations(mat, 2):
            group_probs.append(np.exp(v1+v2).sum())
        mprob = np.mean(group_probs)
        expected_probs_per_group[i] = mprob

    mean_prob_overall = np.mean(expected_probs_per_group)
    squared_loss_overall = (expected_probs_per_group**2).sum()
    sys.stderr.write("%.2f\t%.2f\t%s\n" % (mean_prob_overall, squared_loss_overall, common.pretty_probvector(expected_probs_per_group)))
    return mean_prob_overall, squared_loss_overall, expected_probs_per_group


if __name__ == "__main__":
    pass
