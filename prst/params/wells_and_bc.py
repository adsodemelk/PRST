from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import six

import numpy as np

import prst

class BoundaryCondition:
    """
    Stores boundary conditions for grid.

    There can only be a single boundary condition per face in the grid. This is
    now enforced. Solvers assume boundary conditions are given on the boundary;
    conditions in the interior of the domain yield unpredictable results.

    Synopsis:

        bc = BoundaryCondition(one_face, "pressure", np.array([10000000]))
        bc = BoundaryCondition(three_faces, "flux", np.array([1,2,3]))
        bc.add(np.array([120, 121]), "flux", 5)
        bc.add(faces, "pressure", 1000000)

    Arguments:

      faces(ndarray):
        Global face indices in external model for which this boundary condition
        should be applied. Must have shape (n,).

      type(ndarray or string):
        Type of boundary condition. Supported values are "pressure" and "flux",
        or a 1d array of strings.

      values(ndarray or scalar):
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
    def __init__(self, faces, types, values, sat=None):
        self.face = None
        self.type = None
        self.val = None
        self.sat = None
        self.add(faces, types, values, sat)

    def add(self, faces, types, values, sat=None):
        if len(faces) == 0:
            prst.log.warn("Empty list of boundary faces.")
            return

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
        self.val = _create_or_append(self.val, values)
        if sat is None:
            self.sat = None
        else:
            self.sat = _create_or_vstack(self.sat, sat)

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

