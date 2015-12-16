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

    def test_equal_without_indexMap(self):
        V = loadMRSTGrid("tests/test_gridprocessing/grid_equal_without_indexMap.mat", "V")
        V == V

    def test_equal_except_indexMap(self):
        G = cartGrid(np.array([4, 5]))
        V = cartGrid(np.array([4, 5]))
        assert hasattr(G.cells, "indexMap")
        del G.cells.indexMap
        assert not hasattr(G.cells, "indexMap")
        assert G != V

    def test_equal_except_volumes_del(self):
        G = cartGrid(np.array([4, 5]))
        V = cartGrid(np.array([4, 5]))
        computeGeometry(G)
        computeGeometry(V)
        # Different values
        G.cells.volumes[0] = 1000
        assert G != V
        # Attribute missing from G
        del G.cells.volumes
        assert G != V
        # Wrong shape
        G.cells.volumes = np.zeros((5,5))
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


class TestComputeGeometry:

    def test_compare_MRST_triangleGrid2D(self):
        G_mrst = loadMRSTGrid("tests/test_gridprocessing/computeGeometry_triangleGrid2D_expected.mat")
        G_pyrst = loadMRSTGrid("tests/test_gridprocessing/computeGeometry_triangleGrid2D.mat")
        computeGeometry(G_pyrst)
        assert G_mrst == G_pyrst

    def test_compare_MRST_triangleGrid3D(self):
        G_mrst = loadMRSTGrid("tests/test_gridprocessing/computeGeometry_triangleGrid3D_expected.mat")
        G_pyrst = loadMRSTGrid("tests/test_gridprocessing/computeGeometry_triangleGrid3D.mat")
        computeGeometry(G_pyrst)
        assert G_mrst == G_pyrst

    def test_findNeighbors2D(self):
        G_pyrst = loadMRSTGrid("tests/test_gridprocessing/computeGeometry_findNeighbors2D.mat")
        with pytest.raises(ValueError):
            computeGeometry(G_pyrst)

    def test_findNeighbors3D(self):
        G_pyrst = loadMRSTGrid("tests/test_gridprocessing/computeGeometry_findNeighbors3D.mat")
        computeGeometry(G_pyrst)
        G_mrst = loadMRSTGrid("tests/test_gridprocessing/computeGeometry_findNeighbors3D_expected.mat")
        G_mrst._cmp(G_pyrst)
        assert G_mrst == G_pyrst

    def test_findNeighbors3D_force(self):
        G_mrst = loadMRSTGrid("tests/test_gridprocessing/computeGeometry_findNeighbors3D_expected.mat")
        computeGeometry(G_mrst, findNeighbors=True)

    def test_hingenodes(self):
        G_pyrst = cartGrid(np.array([3, 3, 3]))
        with pytest.raises(NotImplementedError):
            computeGeometry(G_pyrst, hingenodes=True)

    def test_zeroAreaFaces(self):
        G = cartGrid(np.array([3, 3, 3]))
        # Set all node coordinates to the same...
        G.nodes.coords = np.zeros(G.nodes.coords.shape)
        computeGeometry(G)
        # Faces have correct areas -- zero...
        assert np.isclose(np.linalg.norm(G.faces.areas, 2), 0)
        # ...but cells do not. They become NaN, possibly because of division by
        # zero. This is incorrect behavior documented in the test.
        assert np.isnan(G.cells.volumes[0])

    def test_surfaceGrid(self):
        G = cartGrid(np.array([3, 3, 3]))
        G.gridDim = 2
        with pytest.raises(NotImplementedError):
            computeGeometry(G)

    def test_invalidDimensions(self):
        G = cartGrid(np.array([3, 3, 3]))
        G.gridDim = 4
        with pytest.raises(ValueError):
            computeGeometry(G)

    def test_no_gridType(self):
        G = cartGrid(np.array([3, 3, 3]))
        del G.gridType
        computeGeometry(G)



