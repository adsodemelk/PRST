# Python 3 compatability
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import prst
import numpy as np

@six.python_2_unicode_compatible
class Rock:
    """
    Rock structure for storing permeability and porosity values.

    Synopsis:
       rock = Rock(G, poro=0.5)
       rock = Rock(G, poro=np.array([0.1, 0.2, 0.3]))
       rock = Rock(G, perm=np.array([[1, 2, 3],
                                     [2, 3, 4],
                                     ...,
                                     [1, 2, 3]]))

    Arguments:

      - perm(Optional[scalar or ndarray]): Permeability field. Supported input
        types are:
          - A scalar. Interpreted as uniform, scalar (i.e., homogeneous and
            isotropic) permeability field repeated for all active cells in the
            grid G.

          - A row vector of 1/2/3 columns in two space dimensions or 1/3/6
            columns in three space dimensions. The row vector will be repeated
            for each active cell in the grid G and therefore interpreted as a
            uniform (i.e., a homogeneous), possibly anisotropic, permeability
            field. The parameter must be two-dimensional. E.g. `np.array([[1,
            2, 3]])`.

          - A matrix with column count as above, but with G.cells.num rows in
            total. This input will be treated as per-cell values, resulting in
            heterogeneous permeability. E.g. `np.array([[1,2,3], [4,5,6]])`.

      - poro(Optional[scalar or ndarray]): Porosity field. Can be either a
        single, scalar value or an array with one entry per cell. Non-positive
        values will result in a warning. E.g., `np.array([0.1, 0.2, 0.3])`.

      - ntg(Optional[scalar or ndarray]): Net-to-gross factor. Either a single
        scalar value that is repeated for all active cells, or an array with
        one entry per cell. NTG acts as a multiplicative factor on porosity
        when calculating pore volumes. Typically in the range [0 .. 1].
        E.g., `0.5` or `np.array([0.5, 0.5, 0.4])`.

    Example:
    >>> from prst.gridprocessing import cartGrid
    >>> from prst.utils.units import *
    >>> G = cartGrid([10, 20, 30], [1, 1, 1])
    >>> r1 = Rock(G, 100*milli*darcy, 0.3)

    See also:
    NOTE: These do not yet exist in PRST.
        prst.params.rock.poreVolume
        prst.params.rock.permTensor
        prst.solvers.computeTrans
    """
    def __init__(self, G, perm, poro, ntg=None):
        nc = G.cells.num

        # Permeability
        assert np.isscalar(perm) or perm.ndim == 2, \
                "Permeability must be scalar or row vector."
        perm = _expandToCell(perm, nc)
        nt = perm.shape[1]
        if G.gridDim == 2:
            assert nt in [1, 2, 3], \
                    "Permeability must have 1/2/3 columns for "+\
                    "scalar/diagonal/full tensor respectively in 2D. "+\
                    "You supplied %d components" % nt
        else:
            assert nt in [1, 3, 6], \
                    "Permeability must have 1/3/6 columns for "+\
                    "scalar/diagonal/full tensor respectively in 3D. "+\
                    "You supplied %d components" % nt
        self.perm = perm

        # Porosity
        assert np.isscalar(poro) or poro.ndim == 1, \
                "Porosity must be scalar or 1d array."
        poro = _expandToCell(poro, nc)
        if np.any(poro <= 0):
            prst.log.warn("Zero or negative porosity found in cells.")
        self.poro = poro

        # Net-to-gross
        if not ntg is None:
            assert np.isscalar(ntg) or ntg.ndim == 1, \
                    "NTG must be scalar or 1d array."
            self.ntg = _expandTocell(ntg, nc)

    def __str__(self):
        return str(self.__dict__)

def makeRock():
    """Use prst.params.rock.Rock"""
    raise NotImplementedError("Use prst.params.rock.Rock")

def _expandToCell(vals, num_cells):
    if np.isscalar(vals):
        # If scalar, convert to 2d array
        vals = np.array([[vals]])
        return np.tile(vals, (num_cells,1))
    elif vals.ndim == 1:
        # Do nothing if 1d array, just check that dimensions are correct.
        assert vals.shape[0] == num_cells
        return vals
    elif vals.ndim == 2 and vals.shape[0] == 1:
        # Duplicate the row to each cell
        return np.tile(vals, (num_cells,1))
    elif vals.ndim == 2 and vals.shape[0] == num_cells:
        # Do nothing, data was correct
        return vals
    else:
        raise ValueError("Invalid input data dimensions")

def permTensor(rock, dim, rowcols=False):
    """
    Expand permeability tensor to full format.

    Synopsis:
        K = permTensor(rock, dim)
        K, r, c = permTensor(rock, dim, rowcols=True)

    Parameters:
        rock (Rock):
            Rock data structure with valid field `perm`-- one value of the
            permeability tensor for each cell (nc) in the discretised model.

            The field `rock.perm` may have ONE column for a scalar (isotropic)
            permeability in each cell, TWO or THREE columns for a diagonal
            permeability in each cell (in two or three space dimensions,
            respectively) and THREE or SIX columns for a symmetric, full tensor
            permeability. In the latter case, each cell tets the permeability tensor

                K_i = [ k1  k2 ]     in two space dimensions

                K_i = [ k1  k2  k3 ] in three space dimensions.
                      [ k2  k4  k5 ]
                      [ k3  k5  k6 ]

        dim (ndarray):
            Number of space dimensions in the discretised model. Must be two or
            three.

    Returns:
        K: Expanded permeability tensor. An (nc,dim**2) shaped array of
        permeability values as described above.

        r, c: Row- and column indices from which the two form

                (x, Ky) = \sum_r \sum_c x(r) * K(r,c) * y(c)

              may be easily evaluated by a single call to np.sum (see Example).
              OPTIONAL, only returned when `rowcols=True`.

    Example:
        See MRST (params/rock/permTensor.m). DOCTODO.
    """
    if rock is None or not hasattr(rock, "perm"):
        raise ValueError("Rock structure must have attribute `perm`.")

    nc, nk = rock.perm.shape
    if dim == 1:
        prst.log.warn("TODO: Add tests for permTensor dim==1")
        assert nk == 1, \
            "A one dimensional domain does not support multi-component tensors."
        K = rock.perm
        r, c = 0, 0

    elif dim == 2:
        prst.log.warn("TODO: Add tests for permTensor dim==2")

        if nk == 1:
            # Isotropic
            K = np.dot(rock.perm, np.array([[1,0,1]]))

        elif nk == 2:
            # Diagonal
            K = np.column_stack((rock.perm[:,0], np.zeros(nc), rock.perm[:,1]))

        elif nk == 3:
            # Full, symmetric 2-by-2
            K = rock.perm

        else:
            raise ValueError(dim+"-component permeability value is not "+\
                             "supported in two space dimensions")
        K = K[:, [0,1,1,2]]
        r = np.array([0,0,1,1])
        c = np.array([0,1,0,1])

    elif dim == 3:
        if nk == 1:
            # Isotropic
            K = np.dot(rock.perm, np.array([[1,0,0,1,0,1]]))

        elif nk == 3:
            # Diagonal
            K = np.column_stack((
                    rock.perm[:,0],
                    np.zeros(nc),
                    np.zeros(nc),
                    rock.perm[:,1],
                    np.zeros(nc),
                    rock.perm[:,2]
                ))
        elif nk == 6:
            # Full, symmetric, 3-by-3
            K = rock.perm

        else:
            raise ValueError(dim+"-component permeability value is not "+\
                             "supported in three space dimensions")

        K = K[:, [0,1,2,1,3,4,2,4,5]]
        r = np.array([0,0,0,1,1,1,2,2,2])
        c = np.array([0,1,2,0,1,2,0,1,2])

    if rowcols:
        return K, r, c
    else:
        return K

