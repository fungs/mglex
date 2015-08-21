#!/usr/bin/env python3
# takes the number of data points and the number of the class as input and
# generates a matrix with according log-transformed 0/1 entries

from sys import argv, stdout

classindex = int(argv[1])
classnum = int(argv[2])

try:
    datapoints = int(argv[3])
except IndexError:
    datapoints = 1

line = "\t".join(["0.0" if i == classindex else "inf" for i in range(1, classnum+1)]) + "\n"

for i in range(datapoints):
    stdout.write(line)
