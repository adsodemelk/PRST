from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
import numpy as np

import prst
from prst import gridprocessing
from prst.gridprocessing import cartGrid
from prst.params.rock import Rock, permTensor
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
        rock = Rock(G, perm=1, poro=1)

    def test_expandToCell(self):
        # 1d arrays are used a lot in PRST, but in this case
        # it is better to use a 2d array, since there are
        # fewer special cases.
        G = gridprocessing.cartGrid([5, 5, 5])
        rock = Rock(G, perm=0.1, poro=1)
        assert rock.perm.ndim == 2
        assert rock.perm.shape[0] == 5*5*5
        assert rock.perm.shape[1] == 1

    def test_porosity_required(self):
        # default porosity is 1
        G = gridprocessing.cartGrid([3, 3, 3])
        with pytest.raises(TypeError):
            rock = Rock(G, perm=0.1)

class Test_permTensor:
    def setup_method(self, method):
        # Some Rock instances.
        # The permeability will be changed in tests.
        self.rock2 = Rock(cartGrid([4,5]), perm=5, poro=1)
        self.nc2 = 4*5
        self.rock3 = Rock(cartGrid([3,4,5]), perm=5, poro=1)
        self.nc3 = 3*4*5

    def test_permTensor_dim2_perm1(self):
        self.rock2.perm = np.array(
           [[5],
            [5],
            [5],
            [5]])
        K_prst = permTensor(self.rock2, 2)
        K_mrst = np.array(
                [[5,0,0,5],
                 [5,0,0,5],
                 [5,0,0,5],
                 [5,0,0,5]])
        assert np.array_equal(K_prst, K_mrst)

    def test_permTensor_dim2_perm2(self):
        self.rock2.perm = np.array(
           [[5,3],
            [5,3],
            [5,3],
            [5,3]])
        K_prst = permTensor(self.rock2, 2)
        K_mrst = np.array(
                [[5,0,0,3],
                 [5,0,0,3],
                 [5,0,0,3],
                 [5,0,0,3]])
        assert np.array_equal(K_prst, K_mrst)

    def test_permTensor_dim2_perm3(self):
        self.rock2.perm = np.array(
           [[5,3,1],
            [5,3,1],
            [5,3,7],
            [5,3,7]])
        K_prst = permTensor(self.rock2, 2)
        K_mrst = np.array(
                [[5,3,3,1],
                 [5,3,3,1],
                 [5,3,3,7],
                 [5,3,3,7]])
        assert np.array_equal(K_prst, K_mrst)

    def test_permTensor_dim3_perm1(self):
        self.rock3.perm = np.array(
           [[5],
            [5],
            [5],
            [5]])
        K_prst = permTensor(self.rock3, 3)
        K_mrst = np.array(
                [[5,0,0,0,5,0,0,0,5],
                 [5,0,0,0,5,0,0,0,5],
                 [5,0,0,0,5,0,0,0,5],
                 [5,0,0,0,5,0,0,0,5]])
        assert np.array_equal(K_prst, K_mrst)

    def test_permTensor_dim3_perm3(self):
        self.rock3.perm = np.array(
            [[5,3,1],
             [5,3,1],
             [5,3,7],
             [5,3,7]])
        K_prst = permTensor(self.rock3, 3)
        K_mrst = np.array(
                [[5,0,0,0,3,0,0,0,1],
                 [5,0,0,0,3,0,0,0,1],
                 [5,0,0,0,3,0,0,0,7],
                 [5,0,0,0,3,0,0,0,7]])
        assert np.array_equal(K_prst, K_mrst)

    def test_permTensor_dim3_perm6(self):
        self.rock3.perm = np.array(
            [[5,3,1,6,6,6],
             [5,3,1,6,6,6],
             [5,3,7,6,6,6],
             [5,3,7,6,6,6]])
        K_prst = permTensor(self.rock3, 3)
        K_mrst = np.array(
            [[5,3,1,3,6,6,1,6,6],
             [5,3,1,3,6,6,1,6,6],
             [5,3,7,3,6,6,7,6,6],
             [5,3,7,3,6,6,7,6,6]])
        assert np.array_equal(K_prst, K_mrst)

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
        assert bc.face[0] == 120
        assert bc.type[0] == "pressure"
        assert bc.value[0] == 1000000
        assert bc.sat is None




