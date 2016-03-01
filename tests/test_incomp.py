from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

from prst.utils import Struct
from prst.utils.units import *
from prst.incomp.fluid import SingleFluid


class Test_Fluid(object):

    def test_initSingleFluid(self):
        fluid = SingleFluid(viscosity=1*centi*poise, density=1014*kilogram/meter**3)
        mu, rho = fluid.viscosity(), fluid.density()
        assert mu == 1*centi*poise
        assert rho == 1014*kilogram/meter**3
        x = Struct(s=1)
        assert fluid.saturation(x) == 1
