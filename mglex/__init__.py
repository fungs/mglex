# auto-import submodules and subpackages for convenience
from . import common, models, types, evaluation

import pkg_resources  # set global version dynamically by different means
try:
    __version__ = version = pkg_resources.get_distribution('mglex').version  # read version from installation catalog
except pkg_resources.DistributionNotFound:
    try:
        from os.path import dirname, join
        from setuptools_scm import get_version
        __version__ = get_version(join(dirname(__file__), "../"))  # read version from git source
    except ImportError:
        from sys import stderr
        stderr.write("Cannot determine MGLEX package version, install properly "
                     "or install \'setuptools_scm\' when running in GIT project dir.\n")
        __version__ = "UNKNOWN_VERSION"
