#!/usr/bin/env python3

from setuptools import setup, find_packages
import os
import re

# def read_file(fname):
#     return open(os.path.join(os.path.dirname(__file__), fname)).read()
#
#
# def get_version(filename):
#     verstrline = open(filename, "rt").read()
#     VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
#     mo = re.search(VSRE, verstrline, re.M)
#     if mo:
#         return mo.group(1)
#     else:
#         raise RuntimeError("Unable to find version string in %s." % (filename,))
#
# def get_author(filename):
#     pass


setup(
    name='mglex',
    description='mglex - MetaGenome Likelihood Extractor',
    url='http://github.com/fungs/mglex',
    author='Johannes Dröge',
    author_email='johannes.droege@uni-duesseldorf.de',
    license='GNU General Public License, version 3 (GPL-3.0)',
    packages=find_packages(),
    scripts=['build-responsibility', 'classify-likelihood', 'evaluate-classification', 'train-ml'],
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=["numpy >= 1.9.2", "scipy >= 0.16.0", "docopt >= 0.6.2"],
    # long_description=read_file('README.md'),
    classifiers=[
        "Development Status :: 3 - Beta",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ]
)
