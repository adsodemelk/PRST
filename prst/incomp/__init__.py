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

__all__ = ["fluid", "transport", "incompTPFA", "capPressureRHS",
           "computePressureRHS"]
from . import fluid, transport
import prst.incomp.fluid.incompressible
import prst.incomp.fluid.utils

# def computeFacePressure # not used in incompTPFA
# not implemented yet


def capPressureRHS():
    raise NotImplementedError

def computePressureRHS():
    raise NotImplementedError

def incompTPFA(state, G, T, fluid, wells=None, bc=None, src=None,
               LinSolve=None, MatrixOutput=False, verbose=None,
               condition_number=False, gravity=None):
    """
    Solve incompressible flow problem (fluxes/pressure) using TPFA method.

    Synopsis:
        state = incompTPFA(state, G, T, fluid)
        state = incompTPFA(state, G, T, fluid, **kwargs)

    Description:
        This function assembles and solves a (block) system of linear equations
        defining interface fluxes and cell pressures at the next time step in a
        sequential splitting scheme for the reservoir simulation problem
        defined by Darcy's law and a given set of external influences (wells,
        sources, and boundary conditions).

        This function uses a two-point flux approximation (TPFA) method with
        minimal memory consumption within the constraints of operating on a
        fully unstructured polyhedral grid structure.

    Arguments:
        state (Struct):
            Reservoir and well solution structure either properly initialized
            from functions `prst.solvers.initResSol` and
            `prst.solvers.initWellSol` respectively, or the results from a
            previous call to function `incompTPFA` and, possibly, a transport
            solver such as function `prst.incomp.transport.implicitTransport`.

        G (Grid):
            prst.gridprocessing.Grid object.

        T (ndarray):
            Half-transmissibilities as computed by function `computeTrans`.
            Column array.

        fluid (Fluid):
            prst.incomp.fluid.SimpleFluid object.

        wells (Optional[Struct]):
            Well structure as defined by function `addWell. May be None, which
            is interpreted as a model without any wells.

        bc (Optional[BoundaryCondition]):
            prst.params.wells_and_bc.BoundaryCondition object. This structure
            accounts for all external boundary conditions to the reservoir
            flow. May be None, which is interpreted as all external no-flow
            (homogeneous Neumann) conditions.

        src (Optional[object]):
            Explicit source contributions as defined by function
            `prst.params.wells_and_bc.addSource`. May be None, which is
            interpreted as a reservoir model without explicit sources.

        LinSolve (Optional[function]):
            Handle to linear system solver function to which the fully
            assembled system of linear equations will be passed. Assumed to
            support the syntax

                x = LinSolve(A, b)

            in order to solve a system Ax=b of linear equations.
            Default: TODO scipy.sparse.solver used?

        MatrixOutput (Optional[bool]):
            Whether or not to return the final system matrix `A` to the caller
            of function `incompTPFA`. Boolean. Default: False.

        verbose (Optional[bool]):
            Enable output. Default value dependent upon global verbose settings
            of `prst.utils.prstVerbose`.

        condition_number (Optional[bool]):
            Display estimated condition number of linear system. Default: False.

        gravity (Optional[ndarray]):
            The current gravity in vector form.
            Default: prst.gravity (=np.array([0, 0, 9.8066]))

    Returns:
        Updates `state` argument with new values for the fields:
            state.pressure: Pressure values for all cells in the discretised
            reservoir model, `G`.

            state.facePressure: Pressure values for all interfaces in the
            discretised reservoir model, `G`.

            state.flux: Flux across global interfaces corresponding to the rows
            of `G.faces.neighbors`.

            state.A: System matrix. Only returned if specifically requested by
            argument `MatrixOutput`.

            state.wellSol: List of well solution Structs. One for each well in
            the model, with new values for the fields:

                - flux: Perforation fluxes through all perforations for
                  corresponding well. The fluxes are interpreted as injection
                  fluxes, meaning positive values correspond to injection into
                  reservoir while negative values mean production/extraction
                  out of reservoir.

                - pressure: Well bottom-hole pressure.

    Note:
        If there are no external influences, i.e., if all of the structures
        `W`, `bc` and `src` are empty and there are no effects of gravity, then
        the input values `xr` and `xw` are returned unchanged and a warning is
        printed in the command window.

    MRST Example: (TODO: Rewrite in PRST)

        G   = computeGeometry(cartGrid([3, 3, 5]));

        f   = initSingleFluid('mu' , 1*centi*poise, ...
                              'rho', 1000*kilogram/meter^3);
        rock.perm = rand(G.cells.num, 1)*darcy()/100;

        bc  = pside([], G, 'LEFT', 2*barsa);
        src = addSource([], 1, 1);
        W   = verticalWell([], G, rock, 1, G.cartDims(2), []   , ...
                           'Type', 'rate', 'Val', 1*meter^3/day, ...
                           'InnerProduct', 'ip_tpf');
        W   = verticalWell(W, G, rock, G.cartDims(1), G.cartDims(2), [], ...
                           'Type', 'bhp', 'Val', 1*barsa, ...
                           'InnerProduct', 'ip_tpf');

        T   = computeTrans(G, rock);

        state         = initResSol (G, 10);
        state.wellSol = initWellSol(G, 10);

        state = incompTPFA(state, G, T, f, 'bc', bc, 'src', src, ...
                           'wells', W, 'MatrixOutput', true);

        plotCellData(G, state.pressure)

    See also:
        computeTrans, addBC, addSource, addWell, initSingleFluid, initResSol,
        initWellSol.
    """
    pass



def _dynamic_quantities():
    raise NotImplementedError

def _compute_trans():
    raise NotImplementedError
