"""
Compares PRST gravityColumn vs MRST gravityColumn.

See docs/examples/1ph/Gravity column.ipynb for explanation of code.
"""
# For Python 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
from helpers import getpath
from scipy.io import loadmat

class TestGravityColumn:

    def test_gravityColumn(self):
        import numpy as np

        import prst
        import prst.incomp as incomp
        import prst.gridprocessing as gridprocessing
        import prst.utils as utils
        import prst.params as params
        import prst.solvers as solvers
        from prst.utils.units import centi, poise, kilogram, meter, bar, darcy

        prst.gravity_reset()
        G = gridprocessing.cartGrid([1, 1, 30], [1, 1, 30])
        gridprocessing.computeGeometry(G)
        rock = params.rock.Rock(G, perm=0.1*darcy, poro=1)
        fluid = incomp.fluid.SingleFluid(viscosity=1*centi*poise,
                                         density=1014*kilogram/meter**3)
        bc = params.wells_and_bc.BoundaryCondition()
        bc.addPressureSide(G, "top", 100*bar)
        T = solvers.computeTrans(G, rock)
        resSol = solvers.initResSol(G, p0=0.0)
        psol = incomp.incompTPFA(resSol, G, T, fluid, bc=bc)

        # Load MRST results and compare solution pressure, flux, saturation,
        # facePressure.
        matfile = getpath("test_example_gravityColumn/sol.mat")
        msol = loadmat(matfile, squeeze_me=True, struct_as_record=False)["sol"]
        msol_pressure = np.atleast_2d(msol.pressure).transpose()
        msol_flux = np.atleast_2d(msol.flux).transpose()
        msol_s = np.atleast_2d(msol.s).transpose()
        msol_facePressure = np.atleast_2d(msol.facePressure).transpose()

        assert np.allclose(psol.pressure, msol_pressure, rtol=1e-11)
        assert np.allclose(psol.flux, msol_flux, rtol=1e-11)
        assert np.allclose(psol.s, msol_s, rtol=1e-11)
        assert np.allclose(psol.facePressure, msol_facePressure, rtol=1e-11)

