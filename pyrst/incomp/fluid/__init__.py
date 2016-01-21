"""
Routines for evaluating pressure and saturation dependent parameters.

Classes:

    Fluid

Submodules:
    incompressible - Models for various simple, incompressible fluids.
    utils -

"""
__all__ = ["incompressible", "utils"]

class Fluid(object):
    """Fluid structure used in PyRST.


    PyRST's fluid representation is an object containing function handles which
    compute fluid properties (e.g., density or viscosity), fluid phase
    saturations, and phase relative permeability curves. This representation
    supports generic implementations of derived quantities such as mobilities
    or flux functions for a wide range of fluid models.

    Specifically, a Fluid object has the following attributes:

        - properties -- A function for evaluating fluid properties such as
                        density or viscosity. Usage:

                            mu, rho = fluid.properties()
                            mu, rho, extra = fluid.properties(state)

    """ 
    def __init__(self, properties=None, saturation=None, relperm=None):
            self.properties = properties
            self.saturation = saturation
            self.relperm = relperm
