"""
Support for incompressible flow/transport problems

Functions available in MRST, but not yet in PRST:
    capPressureRHS      - Compute capillary pressure contribution to system RHS
    computeFacePressure - Compute face pressure using two-point flux approximation.
    computePressureRHS  - Compute right-hand side contributions to pressure linear system.
    incompTPFA          - Solve incompressible flow problem (fluxes/pressures) using TPFA method.

Submodules:
    fluid - Routines for evaluating pressure and saturation dependent parameters.
    transport - Routines for solving transport/saturation equation.

"""

__all__ = ["fluid", "transport"]

# def capPressureRHS
# def computeFacePressure
# def computePressureRHS
# def incompTPFA
