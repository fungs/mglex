#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains all higher-level functionality required by the EM estimation procedure which is independent of the
actual data and models.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from sys import stderr, stdout
from termcolor import colored
from itertools import count
import common
import numpy as np


logfile = open("em.log", "w")

def get_priors(quantities, responsibilities):
    priors = np.squeeze(np.dot(quantities, responsibilities))  # np.asarray necessary?
    priors /= priors.sum()
    common.assert_probarray(priors)  # TODO: remove
    return priors


def e_step(models, priors, data):
    common.assert_probarray(priors)
    loglike = models.log_likelihood(data)
    loglike = loglike + np.log(priors)  # TODO: why doesn't += work?; TODO: use the prior to give an advantage to large clusters? (Bayesian)
    return common.exp_normalize(loglike), common.total_likelihood(loglike)


def m_step(model, responsibilities, data):  # TODO: weighting of data points in model parameter maximization with
                                             # priors or responsibilities? -> sequence length in data? -> consistent?
    # assert_probmatrix(responsibilities)
    priors = get_priors(data.sizes.T, responsibilities)  # TODO: check correctness
    cmask = np.asarray(priors, dtype=bool)  # determine empty clusters
    if np.any(np.logical_not(cmask)):
        stderr.write("LOG M: The following clusters are removed: %s\n"
                     % ",".join(map(str, np.where(np.logical_not(priors))[0])))
    dimchange = model.maximize_likelihood(responsibilities, data, cmask)
    return model, priors[cmask], dimchange


def em(model, priors, data, responsibilities=None, maxiter=None):
    step_counter = count(1)
    dimchange = False
    loglike = None
    diffsum = .0
    diff = 1.
    padlen_delta = 0
    padlen_ll = 0
    delta_counter = 0

    if responsibilities is not None:  # TODO: check correctness
        model, priors, dimchange = m_step(model, responsibilities, data)

    for i in step_counter:
        print("Cluster priors:", file=logfile)
        common.print_probvector(priors, file=logfile)
        common.newline(file=logfile)

        lloglike = loglike
        responsibilities, loglike = e_step(model, priors, data)

        print("Step %i responsibility of first and last data:" % i, file=logfile)
        common.print_probvector(responsibilities[0, :], file=logfile)
        common.print_probvector(responsibilities[-1, :], file=logfile)
        common.newline(file=logfile)

        if lloglike:
            diff = loglike - lloglike
            if diff >= 0.:
                delta_color = "red"
            else:
                delta_color = "blue"
            stderr.write("LOG EM #: %3i | LL: %s | Δ: %s | mix: %s\n" % (
                i, colored("%i" % loglike, "yellow").rjust(padlen_ll),
                colored("%.2f" % diff, delta_color).rjust(padlen_delta),
                colored(" ".join(["%2.2f" % f for f in sorted(priors, reverse=True)]), "green")))
            diffsum += diff
        else:
            loglike_str = "%i" % loglike
            cclen = len(loglike_str)
            loglike_str = colored(loglike_str, "yellow")
            cclen = len(loglike_str) - cclen
            padlen_ll = len(loglike_str)
            padlen_delta = padlen_ll + 3
            stderr.write("LOG EM #: %3i | LL: %s | Δ: %s | mix: %s\n" % (i, loglike_str, "".rjust(padlen_delta - cclen),
                colored(" ".join(["%2.2f" % f for f in sorted(priors, reverse=True)]), "green")))

        model, priors, dimchange = m_step(model, responsibilities, data)

        if common.approx_equal(diff, .0, precision=10**-3):
            delta_counter += 1
        else:
            delta_counter = 0

        if i == maxiter or priors.size == 1 or delta_counter >= 10:
            # print >>stderr, maxiter, priors.size, delta_counter
            break
    return model, priors, responsibilities


def print_clusters(responsibilities, datanames, clusternames, file=stdout):
    infinity = float("inf")
    if responsibilities.shape[0] > 1:
        for name, res in zip(datanames, np.asarray(responsibilities)):
            (i1, l1), (i2, l2) = argmax(res, n=2)
            if l2 != 0.:
                logdiff = np.log(l1) - np.log(l2)
            else:
                logdiff = infinity
                # stderr.write("Responsibility with only 1 spike? -> %s\n" % ",".join(map(lambda f: "%.2f" % f, res)))
            file.write("%s\t%s\t%s\t%.2f\n" % (name, clusternames[i1], clusternames[i2], logdiff))


def print_responsiblities(responsibilities, datanames, clusternames, file=stdout):
    if responsibilities.shape[0] > 1:
        file.write("#%s\t%s\n" % ("responsibility_matrix", "\t".join(clusternames)))
        for name, res in zip(datanames, np.asarray(responsibilities)):
            file.write("%s\t%s\n" % (name, "\t".join(["%.4f" % f for f in res])))


if __name__ == "__main__":
    pass
