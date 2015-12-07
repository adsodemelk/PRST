from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import pytest
import numpy as np

from pyrst.io import loadMRSTGrid

class TestLoadMRSTGrid:

    def test_single_grid_type(self):
        G = loadMRSTGrid("tests/test_io/expected_tensorGrid2D_1G.mat")
        assert isinstance(G.gridType, set)

    def test_multiple_grid_types(self):
        G = loadMRSTGrid("tests/test_io/multiple_gridtypes.mat")
        assert isinstance(G.gridType, set)

    def test_malformed_gridType(self):
        with pytest.raises(ValueError):
            G = loadMRSTGrid("tests/test_io/malformed_gridType.mat")

    def test_no_indexMap(self):
        G = loadMRSTGrid("tests/test_io/grid_without_indexMap_or_cartDims.mat", "V")
        assert not hasattr(G, "indexMap")

    def test_no_cartDims(self):
        G = loadMRSTGrid("tests/test_io/grid_without_indexMap_or_cartDims.mat", "V")
        assert not hasattr(G, "cartDims")

    def test_tensorGrid2D(self):
        G = loadMRSTGrid("tests/test_io/expected_tensorGrid2D_1G.mat")

        # Check existence of top-level attributes
        assert hasattr(G, "cells")
        assert hasattr(G, "faces")
        assert hasattr(G, "nodes")
        assert hasattr(G, "cartDims")
        assert hasattr(G, "gridType")

        # Cells
        assert hasattr(G.cells, "num")
        assert G.cells.num == 4
        assert np.array_equal(G.cells.facePos, np.array(
            [ 0, 4, 8, 12, 16]))
        assert np.array_equal(G.cells.indexMap, np.array(
            [0, 1, 2, 3]))
        assert np.array_equal(G.cells.faces, np.array([
                [0, 0],
                [6, 2],
                [1, 1],
                [8, 3],
                [1, 0],
                [7, 2],
                [2, 1],
                [9, 3],
                [3, 0],
                [8, 2],
                [4, 1],
                [10, 3],
                [4, 0],
                [9, 2],
                [5, 1],
                [11, 3],
            ]))

        # Faces
        assert G.faces.num == 12
        assert np.array_equal(G.faces.nodePos, np.array([
                 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24
            ]))
        # This array is saved as uint8, and if not carefully loaded, -1 will
        # become 255.
        assert G.faces.neighbors[0,0] == -1
        assert np.array_equal(G.faces.neighbors, np.array([
                [-1, 0],
                [0, 1],
                [1, -1],
                [-1, 2],
                [2, 3],
                [3, -1],
                [-1, 0],
                [-1, 1],
                [0, 2],
                [1, 3],
                [2, -1],
                [3, -1],
            ]))
        # G.faces.tag is not available in PyRST
        assert np.array_equal(G.faces.nodes, np.array([
                0, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 8, 1, 0, 2, 1, 4, 3, 5, 4, 7,
                6, 8, 7
            ]))

        # Nodes
        assert G.nodes.num == 9
        # G.nodes.coords
        assert np.array_equal(G.nodes.coords, np.array([
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ]))

        # Top-level attribues
        # cartDims
        assert G.cartDims[0] == 2 and G.cartDims[1] == 2
        assert "tensorGrid" in G.gridType
        assert G.gridType == set(["tensorGrid"])
        assert G.gridDim == 2

    def test_tensorGrid2D_V(self):
        V = loadMRSTGrid("tests/test_io/expected_tensorGrid2D_1V.mat", "V")
        assert V.cells.num == 4

