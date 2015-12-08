from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
import numpy as np

from pyrst.gridprocessing import *
from pyrst.io import loadMRSTGrid


class TestGrid:

    def test_equal_except_cartDims(self):
        G = cartGrid(np.array([4, 5]))
        V = cartGrid(np.array([4, 5]))
        V.cartDims[0] = 10 # Malformed grid for test
        assert G != V

        del V.cartDims
        assert hasattr(G, "cartDims")
        assert not hasattr(V, "cartDims")
        assert G != V

    def test_unequal_grids_cellDim(self):
        G = cartGrid(np.array([4, 5]))
        V = cartGrid(np.array([4, 6]))
        assert G != V

    def test_cmp_equal(self):
        G = cartGrid(np.array([4, 5]))
        G._cmp(G)

    def test_cmp_unequal(self):
        G = cartGrid(np.array([4, 5]))
        V = cartGrid(np.array([4, 6]))
        G._cmp(V)


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


class TestTensorGrid3D:

    def get_simple_params_3d(self):
        return np.array([0, 1]), np.array([0, 1, 2]), np.array([0.5, 1, 1.5])

    def test_compare_MRST_with_depthz(self):
        # Load grid created in Matlab
        #    G = tensorGrid([1 2 3], [0.5, 1, 1.5], [10, 20], 'depthz', [ 1 2 3; 4 5 6; 7 8 9]);
        G_mrst = loadMRSTGrid("tests/test_gridprocessing/expected_tensorGrid3D_1.mat")
        # Create grid using PyRST
        x = np.array([1, 2, 3])
        y = np.array([0.5, 1, 1.5])
        z = np.array([10, 20])
        depthz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        G_pyrst = tensorGrid(x, y, z, depthz=depthz)
        assert G_mrst == G_pyrst

    def test_compare_MRST_with_depthz(self):
        # Load grid created in Matlab
        #    G = tensorGrid([1 2 3], [0.5, 1, 1.5], [10, 20])
        G_mrst = loadMRSTGrid("tests/test_gridprocessing/expected_tensorGrid3D_2.mat")
        # Create grid using PyRST
        x = np.array([1, 2, 3])
        y = np.array([0.5, 1, 1.5])
        z = np.array([10, 20])
        G_pyrst = tensorGrid(x, y, z)
        assert G_mrst == G_pyrst

    def test_wrongly_sized_depthz(self):
        x, y, z = self.get_simple_params_3d()
        depthz = np.zeros([100, 2])
        with pytest.raises(ValueError):
            G = tensorGrid(x, y, z, depthz=depthz)

    def test_invalid_coords(self):
        x, y, z = self.get_simple_params_3d()
        bad_x, bad_y, bad_z = np.array([0, 2, 1]), np.array([0, 2, 1]), np.array([0, 2, 1])
        with pytest.raises(ValueError):
            G = tensorGrid(bad_x, y, z)
        with pytest.raises(ValueError):
            G = tensorGrid(x, bad_y, z)
        with pytest.raises(ValueError):
            G = tensorGrid(x, y, bad_z)

class TestCartGrid2D:

    def test_nonpositive_cellDim(self):
        with pytest.raises(ValueError):
            G = cartGrid(np.array([0, 3]))

    def test_invalid_dimension(self):
        with pytest.raises(ValueError):
            G = cartGrid(np.array([2]))
        with pytest.raises(ValueError):
            G = cartGrid(np.array([2, 2, 2, 2]))

    def test_compare_MRST_simple(self):
        # G = cartGrid([3 5], [1 1]);
        G_mrst = loadMRSTGrid("tests/test_gridprocessing/cartGrid2D_simple.mat")

        # Using numpy array parameters
        G_pyrst = cartGrid(np.array([3, 5]), np.array([1, 1]))
        assert G_mrst == G_pyrst

class TestCartGrid3D:

    def test_compare_MRST_simple(self):
        # G = cartGrid([3 5 7], [1 1 3]);
        G_mrst = loadMRSTGrid("tests/test_gridprocessing/cartGrid3D_simple.mat")
        G_pyrst = cartGrid(np.array([3, 5, 7]), np.array([1, 1, 3]))
        assert G_mrst == G_pyrst
