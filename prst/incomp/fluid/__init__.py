"""
Routines for evaluating pressure and saturation dependent parameters.

Classes:

    Fluid

Submodules:
    incompressible - Models for various simple, incompressible fluids.
    utils -

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

__all__ = ["incompressible", "utils", "Fluid", "SingleFluid"]

@six.python_2_unicode_compatible
class Fluid(object):
    """Fluid structure used in PRST.


    PRST's fluid representation is an object containing functions which
    compute fluid properties (e.g., density or viscosity), fluid phase
    saturations, and phase relative permeability curves. This representation
    supports generic implementations of derived quantities such as mobilities
    or flux functions for a wide range of fluid models.

    Specifically, a Fluid object has the following methods:

        - viscosity -- Calculates phase viscosity
        - density   -- Calculates fluid phase density
        - saturation -- Calculates fluid phase saturations
        - relperm    -- Calculates relative permeability for each fluid phase.

    Derived fluid classes may contain additional methods which may be needed
    in complex fluid models (e.g., Black Oil).

    If there are multiple PVT/property regions in the reservoir model--for
    instance defined by means of the ECLIPSE keyword 'PVTNUM'--then fluid
    attributes must be lists of functions instead of functions, with one
    function for each property region. In this case each of the functions in
    the list must support the syntax

        mu = viscosity(state, i)

    where `i` is an indicator (a boolean array). The call `someattr(state,i)`
    must compute values only for those elements where `i` is True.

    - relperm
        Function or list of functions (one for each relative permeability
        region, i.e., each saturation function region) for evaluating relative
        permeability curves for each fluid phase. Must support the syntax

            kr = relperm(s, state)
            kr, dkr = relperm(s, state, derivatives=[0,1])
            kr, dkr, d2kr = relperm(s, state, derivatives=[0,1,2])

        The first line computes the relative permeability curves at the current
        saturations, `s`. The second line additionally computes the first
        partial derivatives of these curves with respect to each phase
        saturation. Finally, the third line additionally computes second order
        derivatives. This syntax is optional and only needed in a few specific
        instances (e.g., when using the second order adjoint method).

    """
    def __str__(self):
        return str(self.__dict__)

class SingleFluid(Fluid):
    def __init__(self, viscosity, density):
        super(SingleFluid, self).__init__()
        self.mu = viscosity
        self.rho = density

    def viscosity(self):
       return self.mu

    def density(self):
        return self.rho

    def saturation(self, x):
        return x.s

    def relperm(s, derivatives=None):
        if derivatives is None:
            derivatives = [0]
        if len(derivatives) >= 1:
            assert derivatives[0] == 0
            ret = np.ones(len(s))
        if len(derivatives) >= 2:
            assert derivatives[1] == 1
            ret += (np.zeros(len(s)),)
        if len(derivatives) >= 3:
            assert derivatives[2] == 2
            ret += (np.zeros(len(s)),)
        return ret

