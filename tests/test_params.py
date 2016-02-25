from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
import numpy as np

import prst
from prst import gridprocessing
from prst.params.rock import Rock
from prst.params.wells_and_bc import BoundaryCondition

class TestGravity:

    def test_gravity(self):
        assert len(prst.params.gravity) == 3

    def test_gravity_reset(self):
        prst.params.gravity = np.array([0, 0, 0])
        prst.params.gravity_reset()
        assert np.linalg.norm(prst.params.gravity) > 9

class TestRock:

    def test_init(self):
        G = gridprocessing.cartGrid([10, 10, 10])
        rock = Rock(G)

    def test_expandToCell(self):
        # 1d arrays are used a lot in PRST, but in this case
        # it is better to use a 2d array, since there are
        # fewer special cases.
        G = gridprocessing.cartGrid([5, 5, 5])
        rock = Rock(G, perm=0.1)
        assert rock.perm.ndim == 2
        assert rock.perm.shape[0] == 5*5*5
        assert rock.perm.shape[1] == 1

class TestWellsAndBC:

    def test_simple_BoundaryCondition(self):
        # These input parameters are used in the MRST example gravityColumn.m.
        ix = np.array([120])
        bc = BoundaryCondition(ix, "pressure", 10000000)
        assert bc.face[0] == 120
        assert len(bc.face) == 1
        assert bc.type[0] == "pressure"
        assert bc.value[0] == 10000000

        bc = BoundaryCondition(ix, "flux", -1)
        assert bc.face[0] == 120
        assert len(bc.face) == 1
        assert bc.type[0] == "flux"

    def test_empty_condition(self):
        bc = BoundaryCondition()
        assert bc.face == None
        assert bc.type == None
        assert bc.value == None
        assert bc.sat == None
        bc.add(np.array([120]), "pressure", 99)
        assert bc.value[0] == 99

    def test_gravity_column(self):
        G = gridprocessing.cartGrid([1, 1, 30], [1, 1, 30])
        bc = BoundaryCondition().addPressureSide(G, "top", 1000000)
        assert bc.face[0] == 121
        assert bc.type[0] == "pressure"
        assert bc.value[0] == 1000000
        assert bc.sat is None




