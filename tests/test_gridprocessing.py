from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
import numpy as np

from pyrst.gridprocessing import *
from pyrst.io import loadMRSTGrid

class TestTensorGrid2D:
    def get_simple_params_2d(self):
        return np.array([0, 1]), np.array([0, 1, 2])

    def test_function_exists(self):
        x, y = self.get_simple_params_2d()
        G = tensorGrid(x, y)
        assert G is not None

    def test_gridtype_added(self):
        x, y = self.get_simple_params_2d()
        G = tensorGrid(x, y)
        assert hasattr(G, "gridType")
        assert "tensorGrid" in G.gridType

    def test_gridDim_added(self):
        x, y = self.get_simple_params_2d()
        G = tensorGrid(x, y)
        assert G.gridDim == 2

    def test_invalid_coords(self):
        x, y = self.get_simple_params_2d()
        bad_x, bad_y = np.array([0, 2, 1]), np.array([0, 2, 1])
        with pytest.raises(ValueError):
            G = tensorGrid(bad_x, y)
        with pytest.raises(ValueError):
            G = tensorGrid(x, bad_y)

    def test_compare_MRST_simple(self):
        # Load grid created in Matlab using tensorGrid([0 1 2], [0 1 2])
        G_mrst = loadMRSTGrid("tests/test_gridprocessing/expected_tensorGrid2D_1.mat")

        # Create grid using PyRST
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        G_pyrst = tensorGrid(x, y)

        # Compare all variables
        assert G_mrst == G_pyrst
