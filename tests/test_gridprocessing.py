from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
from helpers import getpath
import numpy as np

from prst.gridprocessing import *
from prst.io import loadMRSTGrid


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
        V = loadMRSTGrid(getpath("test_gridprocessing/grid_equal_without_indexMap.mat"), "V")
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

    def test_str(self):
        G = cartGrid(np.array([4, 5]))
        str(G)

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
        assert G.gridType[-1] == "tensorGrid"

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
        G_mrst = loadMRSTGrid(getpath("test_gridprocessing/expected_tensorGrid2D_1.mat"))

        # Create grid using PRST
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        G_prst = tensorGrid(x, y)

        # Compare all variables
        print(G_mrst.gridType, G_prst.gridType)
        assert G_mrst == G_prst


class TestTensorGrid3D:

    def get_simple_params_3d(self):
        return np.array([0, 1]), np.array([0, 1, 2]), np.array([0.5, 1, 1.5])

    def test_compare_MRST_with_depthz(self):
        # Load grid created in Matlab
        #    G = tensorGrid([1 2 3], [0.5, 1, 1.5], [10, 20], 'depthz', [ 1 2 3; 4 5 6; 7 8 9]);
        G_mrst = loadMRSTGrid(getpath("test_gridprocessing/expected_tensorGrid3D_1.mat"))
        # Create grid using PRST
        x = np.array([1, 2, 3])
        y = np.array([0.5, 1, 1.5])
        z = np.array([10, 20])
        depthz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        G_prst = tensorGrid(x, y, z, depthz=depthz)
        assert G_mrst == G_prst

    def test_compare_MRST_without_depthz(self):
        # Load grid created in Matlab
        #    G = tensorGrid([1 2 3], [0.5, 1, 1.5], [10, 20])
        G_mrst = loadMRSTGrid(getpath("test_gridprocessing/expected_tensorGrid3D_2.mat"))
        # Create grid using PRST
        x = np.array([1, 2, 3])
        y = np.array([0.5, 1, 1.5])
        z = np.array([10, 20])
        G_prst = tensorGrid(x, y, z)
        assert G_mrst == G_prst

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

    def test_accepts_list_and_array_gridDim(self):
        G = cartGrid(np.array([4, 4]))
        G = cartGrid([4, 4])

    def test_accepts_list_and_array_physDim(self):
        G = cartGrid(np.array([4, 4]), np.array([0.4, 0.4]))
        G = cartGrid([4, 4], [0.4, 0.4])

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
        G_mrst = loadMRSTGrid(getpath("test_gridprocessing/cartGrid2D_simple.mat"))

        # Using numpy array parameters
        G_prst = cartGrid(np.array([3, 5]), np.array([1, 1]))
        assert G_mrst == G_prst


class TestCartGrid3D:

    def test_compare_MRST_simple(self):
        # G = cartGrid([3 5 7], [1 1 3]);
        G_mrst = loadMRSTGrid(getpath("test_gridprocessing/cartGrid3D_simple.mat"))
        G_prst = cartGrid(np.array([3, 5, 7]), np.array([1, 1, 3]))
        assert G_mrst == G_prst

    def test_index_kind(self):
        G = cartGrid([3, 5, 7])
        for name, arr in [('facePos', G.cells.facePos),
                          ('indexMap', G.cells.indexMap),
                          ('faces', G.cells.faces),
                          ('neighbors', G.faces.neighbors),
                          ('nodes', G.faces.nodes),
                          ('nodePos', G.faces.nodePos)]:
            # Must be int or unsigned int.
            assert arr.dtype.kind in ('i', 'u'), name + " is not integer type array"


class TestComputeGeometry:

    def test_compare_MRST_triangleGrid2D(self):
        G_mrst = loadMRSTGrid(getpath("test_gridprocessing/computeGeometry_triangleGrid2D_expected.mat"))
        G_prst = loadMRSTGrid(getpath("test_gridprocessing/computeGeometry_triangleGrid2D.mat"))
        computeGeometry(G_prst)
        assert G_mrst == G_prst

    def test_compare_MRST_triangleGrid3D(self):
        G_mrst = loadMRSTGrid(getpath("test_gridprocessing/computeGeometry_triangleGrid3D_expected.mat"))
        G_prst = loadMRSTGrid(getpath("test_gridprocessing/computeGeometry_triangleGrid3D.mat"))
        computeGeometry(G_prst)
        assert G_mrst == G_prst

    def test_findNeighbors2D(self):
        G_prst = loadMRSTGrid(getpath("test_gridprocessing/computeGeometry_findNeighbors2D.mat"))
        with pytest.raises(ValueError):
            computeGeometry(G_prst)

    def test_findNeighbors3D(self):
        G_prst = loadMRSTGrid(getpath("test_gridprocessing/computeGeometry_findNeighbors3D.mat"))
        computeGeometry(G_prst)
        G_mrst = loadMRSTGrid(getpath("test_gridprocessing/computeGeometry_findNeighbors3D_expected.mat"))
        G_mrst._cmp(G_prst)
        assert G_mrst == G_prst

    def test_findNeighbors3D_force(self):
        G_mrst = loadMRSTGrid(getpath("test_gridprocessing/computeGeometry_findNeighbors3D_expected.mat"))
        computeGeometry(G_mrst, findNeighbors=True)

    def test_hingenodes(self):
        G_prst = cartGrid(np.array([3, 3, 3]))
        with pytest.raises(NotImplementedError):
            computeGeometry(G_prst, hingenodes=True)

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



