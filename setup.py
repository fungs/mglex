#!/usr/bin/env python3

from setuptools import setup, find_packages
from os import path


def readme():
    here = path.relpath(path.abspath(path.dirname(__file__)))    
    with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
            return f.read()


setup(
    name='MGLEX',
    description='MGLEX - MetaGenome Likelihood EXtractor',
    long_description=readme(),
    url='https://github.com/fungs/mglex',
    author='Johannes Dröge',
    author_email='code@fungs.de',
    license='GNU General Public License, version 3 (GPL-3.0)',
    packages=find_packages(),
    exclude_package_data = {'': ['.gitignore']},
    scripts=['mglex-cli'],
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=["numpy >= 1.8.2", "scipy >= 0.13.3", "docopt >= 0.6.2"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Operating System :: POSIX :: Linux"
    ]
)
