from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import six

import numpy as np

import prst

@six.python_2_unicode_compatible
class BoundaryCondition(object):
    """
    Stores boundary conditions for grid.

    There can only be a single boundary condition per face in the grid. This is
    now enforced. Solvers assume boundary conditions are given on the boundary;
    conditions in the interior of the domain yield unpredictable results.

    Synopsis:

        bc = BoundaryCondition()
        bc = BoundaryCondition(one_face, "pressure", np.array([10000000]))
        bc = BoundaryCondition(three_faces, "flux", np.array([1,2,3]))
        bc.add(np.array([120, 121]), "flux", 5)
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
        if len(faces) == 0:
            prst.log.warn("Empty list of boundary faces.")
            return self

        # Expand type to all faces
        if isinstance(types, six.string_types):
            types = np.tile(types, len(faces))
        else:
            assert isinstance(types, np.array), \
                    "Types must be string or ndarray of strings"
            assert types.ndim == 1, \
                    "Types must be string or 1d array of strings."


        # Check that all types are either pressure or flux
        types = np.char.lower(types)
        assert np.all(np.logical_or(types == "pressure", types == "flux"))

        # Expand single-element saturations to cover all faces.
        if (not sat is None) and sat.ndim == 1:
            sat = np.tile(sat, (len(faces),1))

        # Validate saturation input has same number of columns as existing
        # saturation array.
        assert (sat is None) or (self.sat.shape[1] == sat.shape[1])

        # Expand single-element values to cover all faces
        if np.isscalar(values):
            values = np.tile(values, (len(faces),))

        # Verify that values and sat are same lenght as faces

        # Verify that boundary condition is not already set
        assert np.intersect1d(self.face, faces).size == 0, \
                "New boundary condition overlaps existing conditions."



        self.face = _create_or_append(self.face, faces)
        self.type = _create_or_append(self.type, types)
        self.value = _create_or_append(self.value, values)
        if sat is None:
            self.sat = None
        else:
            self.sat = _create_or_vstack(self.sat, sat)
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

        assert not ((I0 is None) ^ (I1 is None)) # Both or none must be defined

        ix = boundaryFaceIndices(G, side, I0, I1, range)
        assert np.isscalar(p) or p.size == ix.size
        assert sat is None or (sat.shape[0] == 1 or sat.shape[0] == ix.size)

        if (not sat is None) and sat.shape[0] == 1:
            sat = np.tile(sat, (ix.size,1))

        if np.isscalar(p):
            p = np.tile(p, ix.size)

        self.add(ix, "pressure", p, sat=sat)

        return self

    def __str__(self):
        return str(self.__dict__)

def _create_or_append(existing, new):
    if existing is None:
        return new
    else:
        return np.append(existing, new)

def _create_or_vstack(existing, new):
    if existing is None:
        return new
    else:
        return np.vstack((existing, new))

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
        ix - Required face indices.

    Note:
        This function is mainly intended for internal use in this file. Its
        calling interface may change more frequently than functions in the
        BoundaryCondition class.

    See also:
        prst.params.wells_and_bc.BoundaryCondition
    """
    I0 = I0.ravel()
    I1 = I1.ravel()
    I2 = I2.ravel()

    ## Extract all faces of cells within the given subset.
    cells, ft, isOutF = _boundaryCellsSubset(G, side, I0, I1, I2)

def _boundaryCellsSubset(G, direction, I0, I1, I2):
    # Determine which indices and faces to look for.
    d = direction.lower()
    if d in ["left", "left", "xmin"]:
        d0, d1, d2, faceTag = 1, 2, 0 ,0
    elif d in ["right", "east", "xmax"]:
        d0, d1, d2, faceTag = 1, 2, 0, 1
    elif d in ["back", "south", "ymin"]:
        d0, d1, d2, faceTag = 0, 2, 1, 2
    elif d in ["front", "north", "ymax"]:
        d0, d1, d2, faceTag = 0, 2, 1, 3
    elif d in ["top", "upper", "zmin"]:
        d0, d1, d2, faceTag = 0, 1, 2, 4
    elif d in ["bottom", "lower", "zmax"]:
        d0, d1, d2, faceTag = 0, 1, 2, 5
    else:
        raise ValueError("Boundary side " + d + " not supported.")

    # Determine unique outer cells (i.e., cells on boundary)
    # First, remove cells without external faces.
    # Then, get the unique cells the remaining faces neighbor.
    isOutF = np.any(G.faces.neighbors == -1, axis=1)
    cells = np.unique(np.sum(G.faces.neighbors[isOutF,:], axis=1)+1)

    # Determine logical indices of these cells Assume we will only be called
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
        if len(I2) > 0 && np.any(I2 > 1):
            raise ValueError(
                "A non-zero cell depth is incompatible with a "+\
                "two-dimensional grid model. Specifically `I3` must be empty.")

    1/0
    TODO: http://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python

    1) Open Stackoverflow, MATLAB, Jupyter Notebook
    2) Set a breakpoint on line 166 of boundaryFaceIndices.m
    3) Run gravityColumn.m example.
    4) Replicate ind2sub in PRST
    5) Continue writing _boundaryCellsSubset
    6) Continue writing boundaryFaceIndices
    7) Kjøp cola!





