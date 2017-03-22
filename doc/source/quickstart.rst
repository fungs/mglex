Quickstart guide
================

This description is supposed to give a rough overview of how MGLEX handles data and how the operations are called.

Feature files
-------------

For each submodel in MGLEX, a feature file must be provided. These are simple text files where each line corresponds to a contig. Thus, all input files must have the same length and ordering. Features are the data which represent the contigs. They must be derived from the nucleotide sequences, which are not directly needed when working with MGLEX.

Contig assignment matrices
--------------------------

Contig assignments to genomes are both the input to the training step and the output of the classification step. These assignments are represented as a weight or probability matrix in TAB-separated format. Each line corresponds to a contig, as for the feature files, and each column represents a genome or genome bin. All values in this matrix represent weights between zero and one (e.g. likelihoods) but are written and parsed using the negative natural logarithm. Using weights, contigs can be partially assigned to multiple bins, both for model training and in the output. We also refer to such a matrix as a responsibility matrix.

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

After installation, you can call the command line interface

.. code-block:: sh

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

Formatting the data
-------------------

Please refer to the description in the data section to format the feature files. We assume that you have the the following files in the correct format. Usually, these will be

* a file with contig names, one per line: "contigs.seqname"
* a file with contig lengths, one per line: "contigs.seqlen"
* one file for each submodels, one contig per line

It is important, that the n:sup:`th` line in all file corresponds to the same contig. If features are missing, then this is reflected by an empty line.

Building a responsibility matrix
--------------------------------

For each genome we like to model, we require training data. The canonical way to tell the training command which contig to use for which genome, is by passing a responsibility matrix. Doing so, in theory a contig could be used for more than one genome in the training step, e.g. when modeling similar strains. However, this matrix will usually assign each contig to either zero or just one genome. MGLEX provides a simple command to construct such a responsibility matrix from a text file which lists the contigs which are believed to belong to a single genome. For this text file, here called seeds, you must list the contig names for each genome per line, separated by the space symbol. Thus, when you want to model 20 genomes, this seeds file must contain 20 lines. In the following command, we will give MGLEX the contig names file (see above) via standard input and let it construct the responsibility matrix from the seeds file. Because the matrix is very sparse, we will compress it on the fly.

.. code-block:: sh

   > mglex-cli buildmatrix --seeds contigs.seeds < contigs.seqname | gzip > contigs.training.mat.gz

Training a model
----------------

We now use the features and the responsibility matrix to create a model. We also need to provide the contig lengths so MGLEX can weight the contigs according to their size. We decompress the responsibility file and pass it via the standard input. Suppose, we want to construct a model based on k-mer counts in file contigs.kmc.

.. code-block:: sh

   > zcat contigs.training.mat.gz | mglex-cli train --weight contigs.seqlen --composition contigs.kmc --outmodel contigs.model

Depending on the model type and data, this might take a while because the features need first to be parsed and then the model parameters must be calculated.

Classifying contigs
-------------------

Now, suppose we we have seeded the model with some contigs but want to classify all the other, we can call the classify command to (re-)classify the models. Although the seeding contigs were used to construct the model, they can still end up being assigned to another genome than the original one. The output is again a responsibility matrix, but this time it will be less sparse. It contains a normalized likelihood for each contig and each genome. As this file can be large, we will again compress it using gzip.

.. code-block:: sh

   > mglex-cli classify --model contigs.model --composition contigs.kmc | gzip > contigs.classify.mat.gz

The output matrix can be inspected using zless, for instance.

.. code-block:: sh

   > zless -S contigs.classify.mat.gz

If you want to find the most likely genome for a contig, simply find the lowest score in the corresponding row (because this is the negative log-likelihood).

Advanced usage
--------------

This quickstart guide shows the simplest possibility to use MGLEX. There are more commands and some of them are described in the paper. For instance, to get meaningful soft assignments, it is best to re-normalize the output by fixing the beta parameter.
