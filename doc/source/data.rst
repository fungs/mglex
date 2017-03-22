MGLEX feature files
===================

Each submodel in MGLEX defines its own data. The conventions are to use human-readable text files with one line per contig and spaces as the major delimiter with feature files. These files are usually easy to produce other programs and with the help of command line tools like awk, sed, cut, tr etc.

Read coverage
-------------

For each line, the first number sets the average read coverage in the first sample, the second in the second sample etc. These numbers are separated by spaces, so each line should have the same number of entries and a zero where no read could be mapped. Typically, read coverage is calculated by aligning reads to contigs using a read mapper like BWA or Bowtie2. Then, we can parse the resulting BAM files, for instance using BEDtools, to extract the number of mapping reads for each position. Here is an example using Bowtie2, samtools and BEDtools.

First, map the reads (do this for each sample and use different output files).

.. code-block:: sh

 > bowtie2-build contigs.fna contigs.bowtie2
 > bowtie2 -x contigs.bowtie2 -1 forward.fq.gz -2 reverse.fq.gz |
   samtools view -@ 5 -b - < input.sam | samtools sort -@ 5 - out

Then get the number of read for each contig position.

.. code-block:: sh

 > genomeCoverageBed -ibam out.sorted.bam -g contigs.seqlen -d -split |
   awk 'BEGIN{IFS=OFS=FS="\t"}
   {if($1 == last){ s+=$3; c+=1;}
    else{if(s){print last, s/c; s=$3}; c=1; last=$1}}
    END{print last, s/c}' > out.twocol.cov

The awk code here makes sure that positions with zero reads are reported. Then, you must merge still fill in missing contigs with zero counts, which were not reported here, and make sure that this is done in the right order. Finally, you can merge the samples using paste.

The read coverage feature file works for both, the absolute and the relative count submodels.

Nucleotide composition
----------------------

Again, each line lists the features counts (k-mers) separated by spaces. Each line should have the same numer of entries. We can use any k-mer counting program. Here is an example for `fasta2kmerS <https://github.com/algbioi/kmer_counting>`_ using 5-mers.

.. code-block:: sh

 > zcat contigs.fna.gz |
   fasta2kmersS -i <(cat) -f >(cat) -j 5 -k 5 -s 0 -h 0 -n 0 |
   tr '\t' ' ' > contigs.kmc

Taxonomic annotation
--------------------

The tree-shaped annotation can be generated from any taxonomic assignment program. On each line, a taxonomic paths are specified with corresponding weight, for instance an alignment score, the number of matching positions etc. The model will cope with incorrect annotation, so it is beneficial to generate annotation up the the highest-scoring reference taxon up to the species level. A path consist of strings separated by dots and followed by a colon and the numeric weight, for example 1.2.3:100 or Bacteria.Proteobacteria:5. It makes sense to shorten the strings to something numerical in the feature file to save same space on the disk and in the memory which running MGLEX, although this is not required.
