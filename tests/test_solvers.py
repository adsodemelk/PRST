from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
from helpers import getpath
from scipy.io import loadmat
import numpy as np

from prst.gridprocessing import *
from prst.params.rock import Rock
from prst.utils.units import *
from prst.solvers import computeTrans, initResSol

class TestComputeTrans:

    def test_gravityColumn(self):
        G = cartGrid([1,1,30])
        computeGeometry(G)
        rock = Rock(G, perm=0.1*darcy, poro=1)
        rock.perm[0:G.cells.num//2,:] = 0.2*darcy
        T = computeTrans(G, rock)

        mrst_T = loadmat(getpath("test_solvers/computeTrans_gravityColumn_T.mat"),
                         squeeze_me=True, struct_as_record=False)["T"]

        assert np.array_equal(T.shape, mrst_T.shape)
        assert np.allclose(T, mrst_T)

class TestInitResSol:

    def test_common_usage(self):
        G = cartGrid([2,2,2])
        resSol = initResSol(G, p0=0.5)
        assert hasattr(resSol, "pressure")
        assert hasattr(resSol, "flux")
        assert np.array_equal(resSol.pressure, 0.5*np.ones((G.cells.num,1)))
        assert np.array_equal(resSol.flux, np.zeros((G.faces.num,1)))
        assert np.array_equal(resSol.s, np.zeros((G.cells.num,1)))

