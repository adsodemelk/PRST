from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

__all__ = ["BoundaryCondition"]

import numpy as np

import prst
from prst.utils import mcolon

@six.python_2_unicode_compatible
class BoundaryCondition(object):
    """
    Stores boundary conditions for grid.

    There can only be a single boundary condition per face in the grid. This is
    now enforced. Solvers assume boundary conditions are given on the boundary;
    conditions in the interior of the domain yield unpredictable results.

    Synopsis:

        bc = BoundaryCondition()
        bc = BoundaryCondition(one_face, "pressure", np.array([[10000000]]))
        bc = BoundaryCondition(three_faces, "flux", np.array([[1,2,3]]).T)
        bc.add(np.array([[120, 121]]).T, "flux", 5)
        bc.add(faces, "pressure", 1000000)

    Arguments:

      faces(ndarray):
        Global face indices in external model for which this boundary condition
        should be applied. Must have shape (n,).

      type(1d ndarray or string):
        Type of boundary condition. Supported values are "pressure" and "flux",
        or a 1d array of strings.

      values(1D ndarray or scalar):
        Boundary condition value. Interpreted as a pressure value (in units of
        Pascal) when type=="pressure" and as a flux value (in units of m^3/s)
        when type=="flux". One scalar value for each face in `faces`.

        Note: If type=="flux", the values are interpreted as injection flux. To
        specify an extraction flux (i.e., flux out of the reservoir), the
        caller should provide a negative value.

      sat(Optional[ndarray]):
        Fluid composition of fluid injected across inflow faces. An array of
        shape (n,m) of fluid compositions with `n` being the number of faces in
        `faces` and for m=3, the columns interpreted as 0<->Aqua, 1<->Liquid,
        2<->Vapor.

        This field is for the benefit of transport solvers such as
        `blackoilUpwFE` (only in MRST, not PRST) and will be ignored for
        outflow faces.

        Default value: sat=None (assume single-phase flow).

    Properties:
        - face: External faces for which explicit BCs are provided. 1D-ndarray.
        - type: 1D-array of strings denoting type of BC.
        - value:  Boundary condition values for all faces in `face`. They
          correspond to either pressure or flux values, as given by `bcType`.
        - sat: Fluid composition of fluids passing through inflow faces.

    Methods:

      add(...): Adds boundary conditions to an existing object. The syntax is
      the same as the BoundaryCondition initializer.

    Note:
      For convenience, the pressure and flux parameters may be scalar. The
      value is then used for all faces specified in the faces parameter.
    """
    def __init__(self, faces=None, types=None, values=None, sat=None):

        self.face = None
        self.type = None
        self.value = None
        self.sat = None

        # Allow empty boundary condition
        if faces is None and types is None and values is None:
            return
        else:
            self.add(faces, types, values, sat)

    def add(self, faces, types, values, sat=None):
        faces = np.atleast_2d(faces); assert faces.shape[1] == 1
        types = np.atleast_2d(types); assert types.shape[1] == 1
        values = np.atleast_2d(values); assert values.shape[1] == 1
        if sat:
            sat = np.atleast_2d(sat);

        if len(faces) == 0:
            prst.log.warn("Empty list of boundary faces.")
            return self

        # Expand type to all faces
        if isinstance(types, six.string_types):
            types = np.tile(types, (len(faces), 1))
        else:
            assert isinstance(types, np.ndarray), \
                    "Types must be string or ndarray of strings"
            assert types.ndim == 2, \
                    "Types must be string or 2d column array of strings."


        # Check that all types are either pressure or flux
        types = np.char.lower(types)
        assert np.all(np.logical_or(types == "pressure", types == "flux"))

        # Expand single-element saturations to cover all faces.
        if (not sat is None) and sat.shape[0] == 1:
            sat = np.tile(sat, (len(faces), 1))

        # Validate saturation input has same number of columns as existing
        # saturation array.
        assert (sat is None) or (self.sat.shape[1] == sat.shape[1])

        # Expand single-element values to cover all faces
        if values.size == 1:
            values = np.tile(values, (len(faces), 1))
        if types.shape[0] == 1:
            types = np.tile(types, (len(faces), 1))

        # Verify that boundary condition is not already set
        assert np.intersect1d(self.face, faces).size == 0, \
                "New boundary condition overlaps existing conditions."



        self.face = faces
        self.type = types
        self.value = values
        if sat is None:
            self.sat = None
        else:
            self.sat = sat
        return self

    def addPressureSide(self, G, side, p, I0=None, I1=None, sat=None, range=None):
        """
        Impose pressure boundary conditions on global side.

        Synopsis:
            bc.addPressureSide(G, side, pressure)

        Arguments:
            G (Grid):
                prst.gridprocessing.Grid object. Currently restricted to grids
                produced by functions cartGrid and tensorGrid.

            side (str):
                Global side from which to extract face indices. Must (case
                insensitively) match one of the following groups:

                    0) ["west",  "xmin", "left"  ]
                    1) ["east",  "xmax", "right" ]
                    2) ["south", "ymin", "back"  ]
                    3) ["north", "ymax", "front" ]
                    4) ["upper", "zmin", "top"   ]
                    5) ["lower", "zmax", "bottom"]

                These groups correspond to the cardinal directions mentioned as
                the first alternative in each group.

            p (scalar or ndarray):
                Pressure value in units of Pascal, to be applied to the face.
                Either a scalar or a 1d array of length len(I0)*len(I1).
                If ndarray, it must be a column array.

            I0, I1 (Optional[list or ndarray]):
                Cell index ranges for local (in-plane) axes one and two,
                respectively. If an index range is None, it is interpreted as
                covering the entire corresponding local axis of `side` in the
                grid `G`. The local axes on a side of G are ordered according
                to `X` before `Y`, and `Y` before `Z`.

            sat (Optional[ndarray]):
                Fluid composition of fluid injected across inflow faces. An
                array of shape (n,m) of fluid compositions with `n` being the
                number of individual faces specified by (I1, I2) (i.e. n =
                len(I1)*len(I2) or one). If m=3, the columns of `sat` are
                interpreted as 1 <-> Aqua, 2 <-> Liquid, 3 <-> Vapor.

                This field is for the benefit of transport solvers such as
                `blackoilUpwFE` (only in MRST) and will be ignored for outflow
                faces.

                Default: sat=None (assume single-phase flow)

            range (Optional[ndarray]):
                Restricts the search for outer faces to a subset of the cells
                in the direction perpendicular to that of the face. Example: if
                side="left", one will only search for outer faces in the cells
                with logical indices [range,:,:].

                Default: range=None (do not restrict search).

            Returns: self

            Example: See the "simpleBC" and "simpleSRCandBC" examples in MRST.
                     (Not yet in PRST).

            See also:
                fluxside (only MRST) (will be named addFluxSide in PRST)
                prst.params.wells_and_bc.BoundaryCondition
                solveIncompFlow (only MRST)
                prst.gridprocessing.Grid
        """
        if not hasattr(G, "cartDims"):
            return NotImplementedError("Not implemented for this grid type.")
        p = np.atleast_2d(p); assert p.shape[1] == 1

        assert not ((I0 is None) ^ (I1 is None)) # Both or none must be defined

        ix = boundaryFaceIndices(G, side, I0, I1, range)
        assert ix.shape[1] == 1
        assert p.shape[0] == 1 or p.shape[0] == ix.size
        assert sat is None or (sat.shape[0] == 1 or sat.shape[0] == ix.size)

        if (not sat is None) and sat.shape[0] == 1:
            sat = np.tile(sat, (ix.size, 1))

        if p.shape[0] == 1:
            p = np.tile(p, (ix.size, 1))

        self.add(ix, "pressure", p, sat=sat)

        return self

    def __str__(self):
        return str(self.__dict__)

def boundaryFaceIndices(G, side, I0, I1, I2):
    """
    Retrieve face indices belonging to a subset of global outer faces.

    Synopsis:
        ix = boundaryFaceIndices(G, side, I0, I1, I2)

    Arguments:
        G (Grid):
            pyrst.gridprocessing.Grid

        side (str):
            Global side from which to extract face indices. String. Must (case
            insensitively) match one of six alias groups:

                0) ["west",  "xmin", "left"  ]
                1) ["east",  "xmax", "right" ]
                2) ["south", "ymin", "back"  ]
                3) ["north", "ymax", "front" ]
                4) ["upper", "zmin", "top"   ]
                5) ["lower", "zmax", "bottom"]

            These groups correspond to the cardinal directions mentiond as the
            first alternative in each group.

        I0, I1 (list or ndarray):
            Index ranges for local (in-plane) axes one and two, respectively.
            No index range given (I1 or I2 is None) is interpreted as covering
            the entire corresponding local axis of `side` in the grid `G`. The
            local axes on a side in G are ordered `X` before `Y` before `Z`.

        I2 (list or ndarray):
            Index range for global axis perpendicular to `side`. The primary
            purpose of this parameter is to exclude faces *within* a reservoir
            from being added to the return value `ix`. Such faces typically
            occur in faulted reservoirs where a given face may be considered
            external by virtue of being connected to a single reservoir cell
            only.

    Returns:
        ix - Required face indices as a column array.

    Note:
        This function is mainly intended for internal use in this file. Its
        calling interface may change more frequently than functions in the
        BoundaryCondition class.

    See also:
        prst.params.wells_and_bc.BoundaryCondition
    """
    try:
        I0 = I0.ravel()
    except AttributeError:
        pass
    try:
        I1 = I1.ravel()
    except AttributeError:
        pass
    try:
        I2 = I2.ravel()
    except AttributeError:
        pass

    ## Extract all faces of cells within the given subset.
    cells, faceTag, isOutF = _boundaryCellsSubset(G, side, I0, I1, I2)

    fIX = G.cells.facePos;
    hfIX = mcolon(fIX[cells], fIX[cells+1])
    faces = G.cells.faces[hfIX, 0]
    tags = G.cells.faces[hfIX, 1]
    ix = faces[np.logical_and(isOutF[faces], tags == faceTag)]
    # return as column array
    return ix[:,np.newaxis]

def _boundaryCellsSubset(G, direction, I0, I1, I2):
    # Determine which indices and faces to look for.
    d = direction.lower()
    if d in ["left", "left", "xmin"]:
        # I == min(I)
        d0, d1, d2, faceTag = 1, 2, 0 ,0
    elif d in ["right", "east", "xmax"]:
        # I == max(I)
        d0, d1, d2, faceTag = 1, 2, 0, 1
    elif d in ["back", "south", "ymin"]:
        # J == min(J)
        d0, d1, d2, faceTag = 0, 2, 1, 2
    elif d in ["front", "north", "ymax"]:
        # J == max(J)
        d0, d1, d2, faceTag = 0, 2, 1, 3
    elif d in ["top", "upper", "zmin"]:
        # K == min(K)
        d0, d1, d2, faceTag = 0, 1, 2, 4
    elif d in ["bottom", "lower", "zmax"]:
        # K == max(K)
        d0, d1, d2, faceTag = 0, 1, 2, 5
    else:
        raise ValueError("Boundary side " + d + " not supported.")

    # Determine unique outer cells (i.e., cells on boundary)
    # First, remove cells without external faces.
    # Then, get the unique cells the remaining faces neighbor.
    isOutF = np.any(G.faces.neighbors == -1, axis=1)
    cells = np.unique(np.sum(G.faces.neighbors[isOutF,:], axis=1)+1)

    # Determine logical indices of these cells. Assume we will only be called
    # for logically Cartesian grids for which the fields `G.cartDims` and
    # `G.cells.indexMap` are present.
    dims = G.cartDims
    if G.gridDim == 2:
        dims = np.r_[dims[0:2], 1]

        if faceTag > 3:
            raise ValueError("Boundary side "+side+" is not defined for " +\
                             "two-dimensional grids.")
        if len(I1) > 1:
            raise ValueError(
                "Two-dimensional boundary faces are incompatible with a " +\
                "two-dimensional grid model. Specifically `I1` must contain "+\
                "only a single number.")
        if len(I2) > 0 and np.any(I2 > 1):
            raise ValueError(
                "A non-zero cell depth is incompatible with a "+\
                "two-dimensional grid model. Specifically `I3` must be empty.")

    cI, cJ, cK = np.unravel_index(G.cells.indexMap.ravel()[cells], dims, order="F")
    if I0 is None:
        I0 = np.arange(dims[d0])
    if I1 is None:
        I1 = np.arange(dims[d1])

    # Determine whether or not a given cell is within the required subset
    if np.any(I0 < 0) or np.any(dims[d0] <= I0):
        raise ValueError("Cell range `I0` outside model.")
    if np.any(I1 < 0) or np.any(dims[d1] <= I1):
        raise ValueError("Cell range `I1` outside model.")
    if not I2 is None and (np.any(I2 < 0) or np.any(dims[d2] <= I2)):
        raise ValueError("Cell range `I2` outside model.")

    Ii = np.zeros(G.cells.num, dtype=bool)
    Ij = np.zeros(G.cells.num, dtype=bool)
    Ii[I0] = True
    Ij[I1] = True
    inSubSet = np.logical_and(Ii[(cI,cJ,cK)[d0]], Ij[(cI,cJ,cK)[d1]])

    if not I2 is None:
        Ik = np.zeros(G.cells.num, dtype=bool)
        Ik[I2] = True
        inSubSet = np.logical_and(inSubSet, Ik[(cI,cJ,cK)[d2]])

    # Extract required cells subset
    cells = cells[inSubSet]

    # Transpose into column array
    cells = np.atleast_2d(cells)
    if cells.shape[1] > 1:
        cells = cells.T
    return cells, faceTag, isOutF
