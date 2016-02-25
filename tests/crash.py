import logging
import sys

import prst
from prst.io import loadMRSTGrid

from helpers import getpath

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

print("yah")
print(getpath("test_gridprocessing/grid_equal_without_indexMap.mat"))
V = loadMRSTGrid(getpath("test_gridprocessing/grid_equal_without_indexMap.mat"), "V")
print("duew")
V == V
print("wat")
