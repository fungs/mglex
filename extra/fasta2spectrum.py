#!/usr/bin/env python

u"""
This files reads a FASTA file as input and outputs the corresponding DFT frequency spectrum.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

import numpy as np
import matplotlib.pyplot as plt

def dna2vector(dna):
    bv = np.zeros(len(dna), dtype=np.int8)
    for i, c in enumerate(dna.lower()):
        if c in {"a", "t"}:
            bv[i] = 1
        # elif c in {"g", "c"}:
        #     bv[i] = -1
    return bv


def windows(seq, windowsize, stepsize):
    start = 0
    stop = windowsize
    length = len(seq)
    while stop < length:
        yield seq[start:stop]
        start += stepsize
        stop += stepsize


def magnitude(vec, outsize):
    return np.abs(np.fft.fft(vec, outsize))


def plot_spectrum(mag, out):
    n = len(mag)
    # pow = mag**2
    # norm = pow/pow.sum()  # normalized power spectrum
    norm = mag / mag.sum()

    #t = np.arange(n)
    # plt.subplot(2, 1, 1)
    # plt.plot(t, y, 'k-')
    # plt.xlabel('time')
    # plt.ylabel('amplitude')

    if n <= 12:
        xticks = 1
    else:
        xticks = n/float(24)
    #freq = np.arange(n)
    plt.subplot()
    plt.xticks(np.arange(0, n/2, xticks), rotation=65)
    plt.tick_params(axis="both", which="both")
    # plt.minorticks_on()
    plt.grid(b=True, which="both", axis="both")
    # plt.yscale("log")
    plt.plot(np.arange(n/2), norm[:n/2], 'r-')
    plt.xlabel('period')
    plt.ylabel('magnitude')

    plt.savefig(out)


if __name__ == "__main__":
    from Bio import SeqIO
    from sys import stdin, stdout, stderr, argv

    wsize = 300  # TODO: test different windows sizes
    osize = 24
    ssize = 150
    spec = np.zeros(osize, dtype=np.float64)  # TODO: adjust precision
    c = 0

    records = SeqIO.parse(stdin, "fasta")
    for rec in records:
        vec = dna2vector(rec.seq)
        for win in windows(vec, wsize, ssize):
            spec += magnitude(win, osize)
            c += 1
    spec /= np.float64(c)
    stderr.write("Measured %i individual windows of size %i\n" % (c, wsize))
    plot_spectrum(spec, stdout)
    # print spec
#    plot(spectrum, "spectrum.png")
