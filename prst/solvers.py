from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from prst.utils import rldecode, Struct
from prst.params.rock import permTensor

def computeTrans(G, rock, K_system="xyz", cellCenters=None, cellFaceCenters=None,
        verbose=False):
    """
    Compute transmissibilities for a grid.

    Synopsis:
        T = computeTrans(G, rock)
        T = computeTrans(G, rock, **kwargs)

    Arguments:
        G (Grid):
            prst.gridprocessing.Grid instance.

        rock (Rock):
            prst.params.rock.Rock instance with `perm` attribute. The
            permeability is assumed to be in units of metres squared (m^2).
            Use constant `darcy` from prst.utils.units to convert to m^2, e.g.,

                from prst.utils.units import *
                perm = convert(perm, from_=milli*darcy, to=meter**2)

            if the permeability is provided in units of millidarcies.

            The field rock.perm may have ONE column for a scalar permeability in each cell,
            TWO/THREE columns for a diagonal permeability in each cell (in 2/D
            D) and THREE/SIX columns for a symmetric full tensor permeability.
            In the latter case, each cell gets the permability tensor.

                K_i = [ k1  k2 ]      in two space dimensions
                      [ k2  k3 ]

                K_i = [ k1  k2  k3 ]  in three space dimensions
                      [ k2  k4  k5 ]
                      [ k3  k5  k6 ]

        K_system (Optional[str]):
            The system permeability. Valid values are "xyz" and "loc_xyz".

        cellCenters (Optional[ndarray]):
            Compute transmissibilities based on supplied cellCenters rather
            than default G.cells.centroids. Must have shape (n,2) for 2D and
            (n,3) for 3D.

        cellFaceCenters (Optional[ndarray]):
            Compute transmissibilities based on supplied cellFaceCenters rather
            than default `G.faces.centroids[G.cells.faces[:,0], :]`.

    Returns:
        T: Half-transmissibilities for each local face of each grid cell in the
        grid. The number of half-transmissibilities equals the number of rows
        in `G.cells.faces`. 2D column array.

    Comments:
        PLEASE NOTE: Face normals are assumed to have length equal to the
        corresponding face areas. This property is guaranteed by function
        `computeGeometry`.

    See also:
        computeGeometry, computeMimeticIP (MRST), darcy, permTensor, Rock
    """
    if K_system not in ["xyz", "loc_xyz"]:
        raise TypeError(
            "Specified permeability coordinate system must be a 'xyz' or 'loc_xyz'")

    if verbose:
        print("Computing one-sided transmissibilites.")

    # Vectors from cell centroids to face centroids
    assert G.cells.facePos.ndim == 2, "facePos has wrong dimensions"
    cellNo = rldecode(np.arange(G.cells.num), np.diff(G.cells.facePos, axis=0))
    if cellCenters is None:
        C = G.cells.centroids
    else:
        C = cellCenters
    if cellFaceCenters is None:
        C = G.faces.centroids[G.cells.faces[:,0],:] - C[cellNo,:]
    else:
        C = cellFaceCenters - C[cellNo,:]

    # Normal vectors
    sgn = 2*(cellNo == G.faces.neighbors[G.cells.faces[:,0], 0]) - 1
    N = sgn[:,np.newaxis] * G.faces.normals[G.cells.faces[:,0],:]

    if K_system == "xyz":
        K, i, j = permTensor(rock, G.gridDim, rowcols=True)
        assert K.shape[0] == G.cells.num, \
            "Permeability must be defined in active cells only.\n"+\
            "Got {} tensors, expected {} (== num cells)".format(K.shape[0], G.cells.num)

        # Compute T = C'*K*N / C'*C. Loop based to limit memory use.
        T = np.zeros(cellNo.size)
        for k in range(i.size):
            tmp = C[:,i[k]] * K[cellNo,k] * N[:,j[k]]
            # Handle both 1d and 2d array.
            if tmp.ndim == 1:
                T += tmp
            else:
                T += np.sum(tmp, axis=1)
        T = T / np.sum(C*C, axis=1)

    elif K_system == "loc_xyz":
        if rock.perm.shape[1] == 1:
            rock.perm = np.tile(rock.perm, (1, G.gridDim))
        if rock.perm.shape[1] != G.cartDims.size:
            raise ValueError(
                "Permeability coordinate system `loc_xyz` is only "+\
                "valid for diagonal tensor.")
        assert rock.perm.shape[0] == G.cells.num,\
            "Permeability must be defined in active cells only. "+\
            "Got {} tensors, expected {} == (num cells)".format(rock.perm.shape[0], G.cells.num)

        dim = np.ceil(G.cells.faces[:,1] / 2)
        raise NotImplementedError("Function not finished for K_system='loc_xyz'")
        # See MRST, solvers/computeTrans.m

    else:
        raise ValueError("Unknown permeability coordinate system {}.".format(K_system))

    is_neg = T < 0
    if np.any(is_neg):
        if verbose:
            prst.log.warn("Warning: {} negative transmissibilities. ".format(np.sum(is_neg))+
                          "Replaced by absolute values...")
            T[is_neg] = -T[is_neg]

    return np.atleast_2d(T).transpose()
    # GRDECL not supported yet in PRST

def initResSol(G, p0, s0=0.0):
    """
    Initialize incompressible reservoir solution data structure.

    Synopsis:
        state = initResSol(G, p0)
        state = initResSol(G, p0, s0)

    Arguments:
        G (Grid):
            Grid structure

        p0 (scalar or ndarray):
            Initial reservoir pressure. Scalar or array with shape (G.cells.num,).

    Returns: state (State): Initialized reservoir solution structure with
    attributes:
        - pressure -- One scalar pressure value for each cell in `G`.
        - flux     -- One Darcy flux value for each face in `G`.
        - s        -- Phase saturations for all phases in each cell.

    Remarks:
        In the case of a (G.cells.num, 3)-shaped array of fluid saturations
        `state.s`, the columns are generally interpreted as

            0 <-> Aqua, 1 <-> Liquid, 2 <-> Vapour

        Single pressures (p0) and initial phase saturations (s0) are repeated
        uniformly for all grid cells.

        The initial Darcy flux is zero throughout the reservoir.

    See also:
        initWellSol (MRST only), solveIncompFlow (MRST only)
    """
    p0, s0 = np.atleast_2d(p0, s0)

    nc, nf = G.cells.num, G.faces.num
    if hasattr(G, "nnc") and hasattr(G.nnc, "cells"):
        # Expand the number of interfaces with the number of non-neighboring interfaces
        nf += G.nnc.cells.shape[0]

    if s0.shape[0] == 1:
        s0 = np.tile(s0, (nc,1))
    elif s0.shape[0] != nc:
        raise ValueError(
        "Initial saturation must either be 1-by-np or "+\
        "G.cells.num-by-np")

    if p0.size == 1:
        p0 = np.tile(p0, (nc,1))
    else:
        assert p0.shape[0] == nc

    resSol = Struct(pressure=p0,
                    flux=np.zeros((nf,1)),
                    s=s0)
    return resSol


