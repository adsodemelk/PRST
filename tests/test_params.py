from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
import numpy as np

import pyrst
from pyrst import gridprocessing
from pyrst.params.rock import Rock
from pyrst.params.wells_and_bc import BoundaryCondition

class TestGravity:

    def test_gravity(self):
        assert len(pyrst.params.gravity) == 3

    def test_gravity_reset(self):
        pyrst.params.gravity = np.array([0, 0, 0])
        pyrst.params.gravity_reset()
        assert np.linalg.norm(pyrst.params.gravity) > 9

class TestRock:

    def test_init(self):
        G = gridprocessing.cartGrid([10, 10, 10])
        rock = Rock(G)

class TestWellsAndBC:

    def test_simple_BoundaryCondition(self):
        # These input parameters are used in the MRST example gravityColumn.m.
        ix = np.array([120])
        bc = BoundaryCondition(ix, "pressure", 10000000)
        assert bc.face[0] == 120
        assert len(bc.face) == 1
        assert bc.type[0] == "pressure"
        assert bc.val[0] == 10000000

        bc = BoundaryCondition(ix, "flux", -1)
        assert bc.face[0] == 120
        assert len(bc.face) == 1
        assert bc.type[0] == "flux"

    def test_add_BoundaryCondition(self):
        pass

