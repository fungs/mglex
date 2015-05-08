#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
This file contains all higher-level functionality required by the EM estimation procedure which is independent of the
actual data and models.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from sys import stderr, stdout

from termcolor import colored

from common import *


def get_priors(quantities, responsibilities):
    priors = np.squeeze(np.asarray(quantities * responsibilities))
    priors /= priors.sum()
    assert_probarray(priors)  # TODO: remove
    return priors


def e_step(models, priors, data):
    assert_probarray(priors)
    loglike = models[0].log_likelihood(data[0])  # TODO: switch to UniversalData representation?
    # loglike = models.rescale[0]*models[0].log_likelihood(data[0])
    # impact = [total_likelihood(loglike)]
    for m, d in izip(models[1:], data[1:]):
        m_loglike = m.log_likelihood(d)
        # impact.append(total_likelihood(m_loglike))
        # print >>stderr, "data loglike shape:", m_loglike.shape
        loglike += m_loglike
    # impact = np.array(impact)
    # impact = exp_normalize_1d(impact)
    # stderr.write("LOG E: submodel impact %s\n" % impact)
    loglike = loglike + np.log(priors)  # TODO: why doesn't += work?

    # stderr.write("LOG E: Finished E step\n")
    return exp_normalize(loglike), total_likelihood(loglike)


def m_step(models, responsibilities, data):  # TODO: weighting of data points in model parameter maximization with
                                             # priors or responsibilities? -> sequence length in data? -> consistent?
    # assert_probmatrix(responsibilities)
    priors = get_priors(data.sizes, responsibilities)  # TODO: check correctness
    cmask = np.asarray(priors, dtype=bool)  # determine empty clusters
    if np.any(np.logical_not(cmask)):
        stderr.write(
            "LOG M: The following clusters are removed: %s\n" % ",".join(map(str, np.where(np.logical_not(priors))[0])))
    # assert all(priors)  # TODO: remove
    dimchange = any(map(lambda (m, d): m.maximize_likelihood(responsibilities, d, cmask), izip(models, data)))  # TODO: switch to UniversalData representation?
    # stderr.write("LOG M: %i active mixture components\n" % model.components)
    # stderr.write("LOG M: Finished M step\n")
    return models, priors[cmask], dimchange


def em(models, priors, data, responsibilities=None, maxiter=None):
    step_counter = count(1)
    dimchange = False
    loglike = None
    diffsum = .0
    diff = 1.
    padlen_delta = 0
    padlen_ll = 0
    delta_counter = 0

    if responsibilities is not None:  # TODO: check correctness
        models, priors, dimchange = m_step(models, responsibilities, data)

    for i in step_counter:
        lloglike = loglike
        lpriors = priors
        ldimchange = dimchange

        responsibilities, loglike = e_step(models, priors, data)

        if lloglike:
            diff = loglike - lloglike
            if diff >= 0.:
                delta_color = "red"
            else:
                delta_color = "blue"
            stderr.write("LOG EM #: %3i | LL: %s | Δ: %s | mix: %s\n" % (
            i, colored("%i" % loglike, "yellow").rjust(padlen_ll),
            colored("%.2f" % diff, delta_color).rjust(padlen_delta),
            colored(" ".join(map(lambda f: "%2.2f" % f, sorted(priors, reverse=True))), "green")))
            diffsum += diff
        else:
            loglike_str = "%i" % loglike
            cclen = len(loglike_str)
            loglike_str = colored(loglike_str, "yellow")
            cclen = len(loglike_str) - cclen
            padlen_ll = len(loglike_str)
            padlen_delta = padlen_ll + 3
            stderr.write("LOG EM #: %3i | LL: %s | Δ: %s | mix: %s\n" % (i, loglike_str, "".rjust(padlen_delta - cclen),
                                                                         colored(" ".join(map(lambda f: "%2.2f" % f,
                                                                                              sorted(priors,
                                                                                                     reverse=True))),
                                                                                 "green")))

        models, priors, dimchange = m_step(models, responsibilities, data)
        # stderr.write( "dimchange? %s\n" % dimchange)

        if approx_equal(diff, .0, precision=10**-3):
            delta_counter += 1
        else:
            delta_counter = 0

        if i == maxiter or priors.size == 1 or delta_counter >= 10:
            # print >>stderr, maxiter, priors.size, delta_counter
            break
    return models, priors, responsibilities


def print_clusters(responsibilities, datanames, clusternames, out=stdout):
    infinity = float("inf")
    if responsibilities.shape[0] > 1:
        for name, res in izip(datanames, np.asarray(responsibilities)):
            (i1, l1), (i2, l2) = argmax(res, n=2)
            if l2 != 0.:
                logdiff = np.log(l1) - np.log(l2)
            else:
                logdiff = infinity
                # stderr.write("Responsibility with only 1 spike? -> %s\n" % ",".join(map(lambda f: "%.2f" % f, res)))
            out.write("%s\t%s\t%s\t%.2f\n" % (name, clusternames[i1], clusternames[i2], logdiff))


def print_responsiblities(responsibilities, datanames, clusternames, out=stdout):
    if responsibilities.shape[0] > 1:
        out.write("#%s\t%s\n" % ("responsibility_matrix", "\t".join(clusternames)))
        for name, res in izip(datanames, np.asarray(responsibilities)):
            out.write("%s\t%s\n" % (name, "\t".join(map(lambda f: "%.4f" % f, res))))


if __name__ == "__main__":
    pass