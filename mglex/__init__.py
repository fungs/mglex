# auto-import submodules and subpackages for convenience
from . import common, models, types, evaluation

import pkg_resources  # set global version
__version__ = version = pkg_resources.get_distribution('mglex').version
