#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='mglex',
    description='mglex - MetaGenome Likelihood Extractor',
    url='http://github.com/fungs/mglex',
    author='Johannes Dröge',
    author_email='johannes.droege@uni-duesseldorf.de',
    license='GNU General Public License, version 3 (GPL-3.0)',
    packages=find_packages(),
    exclude_package_data = {'': ['.gitignore']},
    scripts=['mglex-cli'],
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
