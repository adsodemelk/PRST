# -*- coding: utf-8 -*-
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import time

import numpy as np
import scipy.sparse.linalg
from numpy_groupies.aggregate_numpy import aggregate

import prst.utils as utils

import prst
import prst.incomp.fluid
import prst.incomp.transport
import prst.incomp.fluid.incompressible
import prst.incomp.fluid.utils



__all__ = ["fluid", "transport", "incompTPFA", "capPressureRHS",
           "computePressureRHS"]

def capPressureRHS():
    raise NotImplementedError("MRST only")

def computePressureRHS(G, omega, bc=None, src=None):
    """
    %Compute right-hand side contributions to pressure linear system.

    TODO: Fix documentation for PRST.
    %
    % SYNOPSIS:
    %   [f, g, h, grav, dF, dC] = computePressureRHS(G, omega, bc, src)
    %
    % DESCRIPTION:
    %   The contributions to the right-hand side for mimetic, two-point and
    %   multi-point discretisations of the equations for pressure and total
    %   velocity
    %
    %     v + lam KÂ·grad (p - gÂ·z omega) = 0
    %     div v  = q
    %
    %   with
    %             __
    %             \    kr_i/
    %     lam   = /       / mu_i
    %             -- i
    %             __
    %             \
    %     omega = /    f_iÂ·rho_i
    %             -- i
    %
    % PARAMETERS:
    %   G     - Grid data structure.
    %
    %   omega - Accumulated phase densities \rho_i weighted by fractional flow
    %           functions f_i -- i.e., omega = \sum_i \rho_i f_i.  One scalar
    %           value for each cell in the discretised reservoir model, G.
    %
    %   bc    - Boundary condition structure as defined by function 'addBC'.
    %           This structure accounts for all external boundary conditions to
    %           the reservoir flow.  May be empty (i.e., bc = struct([])) which
    %           is interpreted as all external no-flow (homogeneous Neumann)
    %           conditions.
    %
    %   src   - Explicit source contributions as defined by function
    %           'addSource'.  May be empty (i.e., src = struct([])) which is
    %           interpreted as a reservoir model without explicit sources.
    %
    % RETURNS:
    %   f, g, h - Pressure (f), source/sink (g), and flux (h) external
    %             conditions.  In a problem without effects of gravity, these
    %             values may be passed directly on to linear system solvers
    %             such as 'schurComplementSymm'.
    %
    %   grav    - Pressure contributions from gravity,
    %
    %                grav = omegaÂ·gÂ·(x_face - x_cell)
    %
    %             where
    %
    %                omega = \sum_i f_i\rho_i,
    %
    %             thus grav is a vector with one scalar value for each
    %             half-face in the model (size(G.cells.faces,1)).
    %
    %   dF, dC  - Dirichlet/pressure condition structure.  Logical array 'dF'
    %             is true for those faces that have prescribed pressures, while
    %             the corresponding prescribed pressure values are listed in
    %             'dC'.  The number of elements in 'dC' is SUM(DOUBLE(dF)).
    %
    %             This structure may be used to eliminate known face pressures
    %             from the linear system before calling a system solver (e.g.,
    %             'schurComplementSymm').
    %
    % SEE ALSO:
    %   addBC, addSource, computeMimeticIP, schurComplementSymm.
    """
    if hasattr(G, "grav_pressure"):
        gp = G.grav_pressure(G, omega)
    else:
        gp = _grav_pressure(G, omega)

    ff = np.zeros(gp.shape)
    gg = np.zeros((G.cells.num, 1))
    hh = np.zeros((G.faces.num, 1))

    # Source terms
    if not src is None:
        prst.warning("computePressureRHS is untested for src != None")
        # Compatability check of cell numbers for source terms
        assert np.max(src.cell) < G.cells.num and np.min(src.cell) >= 0, \
            "Source terms refer to cell not existant in grid."

        # Sum source terms inside each cell and add to rhs
        ss = aggregate(src.cell, src.rate)
        ii = aggregate(src.cell, 1) > 0
        gg[ii] += ss[ii]

    dF = np.zeros((G.faces.num, 1), dtype=bool)
    dC = None

    if not bc is None:
        # Check that bc and G are compatible
        assert np.max(bc.face) < G.faces.num and np.min(bc.face) >= 0, \
            "Boundary condition refers to face not existant in grid."
        assert np.all(aggregate(bc.face, 1)) <= 1, \
            "There are repeated faces in boundary condition."

        # Pressure (Dirichlet) boundary conditions.
        # 1) Extract the faces marked as defining pressure conditions.
        #    Define a local numbering (map) of the face indices to the
        #    pressure condition values.
        is_press = bc.type == "pressure"
        face = bc.face[is_press]
        dC = bc.value[is_press]
        map = scipy.sparse.csc_matrix( (np.arange(face.size),
                                     (face.ravel(), np.zeros(face.size)))  )

        # 2) For purpose of (mimetic) pressure solvers, mark the "face"s as
        #    having pressure boundary conditions. This information will be used
        #    to eliminate known pressures from the resulting system of linear
        #    equations. See e.g. `solveIncompFlow` in MRST.
        dF[face] = True

        # 3) Enter Dirichlet conditions into system right hand side.
        #    Relies implicitly on boundary faces being mentioned exactly once
        #    in G.cells.faces[:,0].
        i = dF[G.cells.faces[:,0],:]
        ff[i] = -dC[map[ G.cells.faces[i[:,0],0],0].toarray().ravel()]

        # 4) Reorder Dirichlet conditions according to sort(face).
        #    This allows the caller to issue statements such as 
        #    `X[dF] = dC` even when dF is boolean.
        dC = dC[map[dF[:,0],0].toarray().ravel()]

        # Flux (Neumann) boundary conditions.
        # Note negative sign due to bc.value representing INJECTION flux.
        is_flux = bc.type == "flux"
        hh[bc.face[is_flux],0] = -bc.value[is_flux]

    if not dC is None:
        assert not np.any(dC < 0)

    return ff, gg, hh, gp, dF, dC

def _grav_pressure(G, omega):
    """Computes innerproduct cf (face_centroid - cell_centroid) * g for each face"""
    g_vec = prst.gravity
    if np.linalg.norm(g_vec[:G.gridDim]) > 0:
        dim = G.gridDim
        assert 1 <= dim <= 3, "Wrong grid dimension"

        cellno = utils.rldecode(np.arange(G.cells.num), np.diff(G.cells.facePos, axis=0))
        cvec = G.faces.centroids[G.cells.faces[:,0],:] - G.cells.centroids[cellno,:]
        ff = omega[cellno] * np.dot(cvec, g_vec[:dim].reshape([3,1]))
    else:
        ff = np.zeros([G.cells.faces.shape[0], 1])
    return ff

def incompTPFA(state, G, T, fluid, wells=None, bc=None, bcp=None, src=None,
               LinSolve=None, MatrixOutput=False, verbose=None,
               condition_number=False, gravity=None,
               pc_form="nonwetting", use_trans=False):
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
        Updates and returns `state` argument with new values for the fields:
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
    if wells is None:
        wells = []
    if LinSolve is None:
        LinSolve = scipy.sparse.linalg.spsolve
    if verbose is None:
        verbose = prst.verbosity
    if gravity is None:
        gravity = prst.gravity


    g_vec = gravity
    # If gravity is overridden, we cannot say anything about the effects of gravity on rhs.
    grav = np.linalg.norm(g_vec[0:G.gridDim]) > 0 or hasattr(G, 'grav_pressure')

    if all([not MatrixOutput, 
            bc is None,
            src is None,
            bcp is None,
            wells is None,
            not grav]):
        prst.log.warn("No external driving forces present in model, state remains unchanged.")

    if verbose:
        print("Setting up linear system...")
        t0 = time.time()

    # Preliminaries
    neighborship, n_isnnc = utils.gridtools.getNeighborship(G, "Topological", True, nargout=2)
    cellNo, cf, cn_isnnc = utils.gridtools.getCellNoFaces(G) # TODO
    nif = neighborship.shape[0]
    ncf = cf.shape[0]
    nc = G.cells.num
    nw = len(wells)
    n = nc + nw

    mob, omega, rho = _dynamic_quantities(state, fluid)
    totmob = np.sum(mob, axis=1, keepdims=True)

    # Compute effective (mobility-weighted) transmissibilities
    T, ft = _compute_trans(G, T, cellNo, cf, neighborship, totmob, use_trans=False)

    # Identify internal faces
    i = np.all(neighborship != -1, axis=1)

    # Boundary conditions and source terms
    hh = np.zeros((nif, 1))
    dF = np.zeros((nif, 1), dtype=bool)
    grav, ff = np.zeros((ncf, 1)), np.zeros((ncf, 1))
    cn_notnnc = np.logical_not(cn_isnnc)[:,0]
    n_notnnc = np.logical_not(n_isnnc)[:,0]
    ff[cn_notnnc,:], gg, hh[n_notnnc,:], grav[cn_notnnc,:], dF[n_notnnc,:], dC = computePressureRHS(G, omega, bc, src)

    # Made to add capillary pressure
    if hasattr(fluid, "pc"):
        prst.warning("incompTPFA for fluid with capillary pressure is UNTESTED. Compare with MRST.")
        pc = fluid.pc(state)
        gpc = np.zeros(totmob.shape)

        if hasattr(fluid, "gpc") and pc_form == "global":
            cc = capPressureRHS(G, mob, pc, gpc, pc_form)
        else:
            cc = capPressureRHS(G, mob, pc, pc_form)
        grav += cc

    sgn = 2*(neighborship[cf,0] == cellNo).astype(np.int)-1
    j = np.logical_or( i[cf[:,0]],  dF[cf[:,0],0] )
    fg = aggregate(cf[j,0], grav[j,0] * sgn[j,0], size=nif)[:,np.newaxis]

    if not bcp is None:
        prst.warning("Code not tested in PRST, cross check with MRST.")
        fg[bcp.face[:,0],0] = fg[bcp.face,0] + bcp.value
        prst.warning("Face pressures are not well defined for periodic boundary faces.")
        if np.any(G.faces.neighbors[:,0] == G.faces.neighbors[:,1]):
            raise ValueError("Periodic boundary: This code does not work of a face is in and outflo.")

    rhs = aggregate(cellNo[:,0], -ft[cf[:,0]] * (sgn[:,0]*fg[cf[:,0],0]+ff[:,0]), size=n) + \
            np.r_[gg[:,0], np.zeros(nw)] + \
          aggregate(cellNo[:,0], -hh[cf[:,0],0], size=n)
    rhs = rhs[:,np.newaxis] # as column vector

    d = np.zeros((G.cells.num, 1))


    if nw:
        raise NotImplementedError("Not yet working with wells. See MRST.")

    """
       # Missing code, from MRST
       % Wells --------------------------
       C    = cell (nw, 1);
       D    = zeros(nw, 1);
       W    = opt.wells;

       for k = 1 : nw,
          wc       = W(k).cells;
          nwc      = numel(wc);
          w        = k + nc;

          wi       = W(k).WI .* totmob(wc);

          dp       = norm(gravity()) * W(k).dZ*sum(rho .* W(k).compi, 2);
          d   (wc) = d   (wc) + wi;

          if     strcmpi(W(k).type, 'bhp'),
             ww=max(wi);
             %ww=1.0;
             rhs (w)  = rhs (w)  + ww*W(k).val;
             rhs (wc) = rhs (wc) + wi.*(W(k).val + dp);
             C{k}     = -sparse(1, nc);
             D(k)     = ww;

          elseif strcmpi(W(k).type, 'rate'),
             rhs (w)  = rhs (w)  + W(k).val;
             rhs (wc) = rhs (wc) + wi.*dp;

             C{k}     =-sparse(ones(nwc, 1), wc, wi, 1, nc);
             D(k)     = sum(wi);

             rhs (w)  = rhs (w) - wi.'*dp;

          else
             error('Unsupported well type.');
          end
       end

       C = vertcat(C{:});
       D = spdiags(D, 0, nw, nw);
    """

    # Add up internal face transmissibilities plus Dirichlet pressure faces for each cell.
    d[:,0] += aggregate(cellNo[dF[cf[:,0],0],0], T[dF[cf,0]], size=nc) + \
         aggregate(neighborship[i,:].ravel(order="F"), np.tile(ft[i], (2,)), size=nc)

    # Assemble coefficient matrix for internal faces. 
    # Boundary conditions may introduce additional diagonal entries. 
    # Also, wells introduce additional equations and unknowns.
    I = np.r_[neighborship[i,0], neighborship[i,1], np.arange(nc)]
    J = np.r_[neighborship[i,1], neighborship[i,0], np.arange(nc)]
    V = np.r_[-ft[i], -ft[i], d[:,0]][:,np.newaxis]
    A = scipy.sparse.csc_matrix((V[:,0], (I, J)), shape=(nc,nc))

    if nw:
        raise NotImplementedError("Wells not implemented yet in PRST.")
    else:
        pass

    if prst.verbosity:
        print("Solving linear system...")
        t0 = time.time()

    if (not np.any(dF)) and (W is None or not np.any(W.type == "bhp")):
        if A[0,0] > 0:
            A[0,0] *= 2
        else:
            j = np.argmax(A.diagonal())
            A[j,j] *= 2

    if condition_number:
        print("*"*60)
        prst.warning("Warning: Condition number estimation method may be slow.")
        print("Condition number is... ", end="")
        import pyamg.util.linalg
        print(pyamg.util.linalg.condest(A))

    if MatrixOutput:
        state.A = A
        state.rhs = rhs

    p = LinSolve(A, rhs)[:,np.newaxis]

    if prst.verbosity:
        print("{} seconds to solve".format(time.time() - t0))

        print("Computing fluxes, face pressures etc...")
        t0 = time.time()

    # Reconstruct face pressures and fluxes
    fpress = (aggregate(cf[:,0], (p[cellNo[:,0],0]+grav[:,0])*T[:,0], size=nif)
              /aggregate(cf[:,0], T[:,0], size=nif))[:,np.newaxis]

    # Neumann faces
    b = np.any(G.faces.neighbors==-1, axis=1)[:,np.newaxis]
    fpress[b[:,0],0] -= hh[b] / ft[b[:,0]]

    # Dirichlet faces
    fpress[dF] = dC

    # Sign for boundary faces
    noti = np.logical_not(i)
    sgn = 2*(G.faces.neighbors[noti,1]==-1)-1
    ni = neighborship[i]
    # Because of floating point loss of precision due to subtraction of similarly sized numbers,
    # this result can be slightly different from MATLAB for very low flux.
    flux = -aggregate(np.where(i)[0], ft[i] * (p[ni[:,1],0]-p[ni[:,0],0]-fg[i,0]), size=nif)[:,np.newaxis]
    c = np.max(G.faces.neighbors[noti,:], axis=1)[:,np.newaxis]
    fg = aggregate(cf[:,0], grav[:,0], size=nif)
    flux[noti,0] = -sgn*ft[noti] * ( fpress[noti,0] - p[c[:,0],0] - fg[noti] )
    state.pressure[0:nc] = p[0:nc]
    state.flux = flux
    state.facePressure = fpress

    if nw:
        raise NotImplementedError("Wells not yet in PRST, only MRST.")
    """
     for k = 1 : nw,
          wc       = W(k).cells;
          dp       = norm(gravity()) * W(k).dZ*sum(rho .* W(k).compi, 2);
          state.wellSol(k).flux     = W(k).WI.*totmob(wc).*(p(nc+k) + dp - p(wc));
          state.wellSol(k).pressure = p(nc + k);
       end
    """
    if prst.verbosity:
        print("{} seconds to compute fluxes, face pressures, etc...".format(time.time()-t0))

    return state

def _dynamic_quantities(state, fluid):
    mu, rho = fluid.viscosity(), fluid.density()
    s = fluid.saturation(state)
    kr = fluid.relperm(s)

    mob = kr / mu
    totmob = np.sum(mob, axis=1, keepdims=True)
    omega = np.sum(mob*rho, axis=1, keepdims=True) / totmob
    return mob, omega, rho

def _compute_trans(G, T, cellNo, cellFaces, neighborship, totmob, use_trans):
    niface = neighborship.shape[0]
    if use_trans:
        neighborcount = np.sum(neighborship != -1, axis=1, keepdims=True)
        assert T.shape[0] == niface, \
            "Expected one transmissibility for each interface " + \
            "(={}) but got {}".format(niface, T.shape[0])
        raise NotImplementedError("Function not yet implemented for use_trans=True. See source code.")
        # Matlab code for rest of function, from mrst-2015b\modules\incomp\incompTPFA.m
        #fmob = accumarray(cellFaces, totmob(cellNo), ...
        #                    [niface, 1]);
        #
        #fmob = fmob ./ neighborcount;
        #ft   = T .* fmob;
        #
        #% Synthetic one-sided transmissibilities.
        #th = ft .* neighborcount;
        #T  = th(cellFaces(:,1));
    else:
        # Define face transmissibility as harmonic average of mobility 
        # weighted one-sided transmissibilities.
        assert T.shape[0] == cellNo.shape[0], \
            "Expected one one-sided transmissibility for each " +\
            "half face (={}), but got {}.".format(cellNo.shape[0], T.shape[0])

        T = T * totmob[cellNo[:,0],:]
        from numpy_groupies.aggregate_numpy import aggregate
        ft = 1/aggregate(cellFaces[:,0], 1/T[:,0], size=niface)
    return T, ft
