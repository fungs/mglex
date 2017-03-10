Quickstart guide
================

This description is supposed to give a rough overview of how MGLEX handles data and how the operations are called.

Feature files
-------------

For each submodel in MGLEX, a feature file must be provided. These are simple text files where each line corresponds to a contig. Thus, all input files must have the same length and ordering. Features are the data which represent the contigs. They must be derived from the nucleotide sequences, which are not directly needed when working with MGLEX.

Contig assignment matrices
--------------------------

Contig assignments to genomes are both the input to the training step and the output of the classification step. These assignments are represented as a weight or probability matrix in TAB-separated format. Each line corresponds to a contig, as for the feature files, and each column represents a genome or genome bin. All values in this matrix represent weights between zero and one (e.g. likelihoods) but are written and parsed using the negative natural logarithm. Using weights, contigs can be partially assigned to multiple bins, both for model training and in the output.

Data handling
-------------

Most data files with MGLEX, in particular the assignment matrices, compress very well. Therefore, we recommend to compress and decompress these files on-the-fly using, for instance, gzip. This will reduce the IO and storage requirements.

MGLEX classification
--------------------

The typical workflow when using MLGEX as a classifier is to

1. get all features in the right text format
2. build an assignment matrix for training
3. train and save a model to a file
4. classify the same of different contigs using this model

MGLEX command line interface
----------------------------

After installation, you can call the command line interface::

   > mglex-cli --help

   This is the mglex command line interface which executes the sub-commands.

   Usage:
    mglex-cli [--version] [--help] <command> [<args>...]

    -h, --help         Show this screen
    -v, --version      Show version

    Here are the commands to run:
     train             Train a model (set maximum-likelihood parameters)
     classify          Calculate likelihood of data under a model
     buildmatrix       Construct a responsibility matrix for grouped data
     evaluate          Evaluate classifications using a reference (true) responsibility matrix
     significance      Give a null model log-likelihood distribution and calculate p-values for unseen data
     bincompare        Compare bins by likelihood values

    See 'mglex-cli <command> --help' for more information on a specific command.
