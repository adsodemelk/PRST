from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import numpy as np
import scipy
import scipy.sparse
from numpy_groupies.aggregate_numpy import aggregate
import numba

import prst.utils
from prst import utils

__all__ = ["Grid", "tensorGrid", "cartGrid", "computeGeometry"]

@six.python_2_unicode_compatible
class Grid(object):
    """
    Grid class used in Python Reservoir Simulation Toolbox.

    Synopsis:
        1) Construct Cartesian grid
            G = cartGrid(...)
            computeGeometry(G)

    Attributes:
        Let G represent a grid in an unstructured format.
        It has the following top level attributes:

        - G.cells (object)
            A structure specifying properties for each individual cell in the
            grid. See CELLS below for details.

        - G.faces (object)
            A structure specifying properties for each individual face in the
            grid. See FACES below for details.

        - G.nodes (object)
            A structure specifying properties for each individual node (vertex)
            in the grid. See NODES below for details.

        - G.gridType (list)
            A list of strings describing the history of grid constructor and
            modifier functions applied to the grid object. Example:

                >>> G = cartGrid(np.array([4, 4, 4]))
                >>> computeGeometry(G)
                <prst.gridprocessing.Grid object at ...
                >>> G.gridType == ["tensorGrid", "cartGrid", "computeGeometry"]
                True

        - G.gridDim (int)
            The dimension of the grid, which in most cases will equal the
            number of coordinate dimensions: G.nodes.coords.shape[1].


        CELLS: Cell structure G.cells has the following attributes:

            - G.cells.num (int)
                Number of cells in the global grid.

            - G.cells.facePos (np.ndarray)
                Array with shape (G.cells.num, 1) into the G.cells.faces array.
                Specifically, the face information of cell `i` is found in the
                subarray

                    G.cells.faces[G.cells.facePos[i] : G.cells.facePos[i+1]]

                The number of faces for each cell may be computed using using
                the statement `np.diff(G.cells.facePos, axis=0)`.

            - G.cells.faces (np.ndarray)
                Array with shape (G.cells.num, 2) of global faces connected to
                a given cell. Specifically, if `G.cells.faces[i,0] == j` then
                global face `G.cells.faces[i,1]` is connected to global cell
                `j`.

                To conserve memory, only the second column is actually stored
                in the grid structure. The first column may be re-constructed
                using the statement

                    prst.util.rldecode(np.arange(G.cells.num),
                        np.diff(G.cells.facePos, axis=0))
                    TODO: Add test for this.

                A grid constructor may, optionally, append a third column to
                this array. In this case `cells.faces[i,2]` often contains a
                tag by which the cardinal direction of face
                `G.cells.faces[i,1]` within cell `cells.faces[i,0]` may be
                distinguished. Some ancillary utilites within the toolbox
                depend on this specific semantic of `G.cells.faces[i,2]`, e.g.
                to easily specify boundary conditions (functions `pside` and
                `fluxside`). NOTE: These are not yet implemented in PRST.

            - G.cells.indexMap (np.ndarray)
                Column array. Maps internal to external grid cells (i.e.,
                active cell numbers to global cell numbers). In the case of
                Cartesian grids, `G.cells.indexMap` is equal to
                `np.arange(G.cells.num)` transposed.

                For grids with a logically Cartesian topology of dimension
                `dims` (a curvilinear grid, a corner-point grid, etc.), a map
                of cell numbers to logical indices may be constructed using the
                following statement in 2D

                    % TODO: Create PRST examples

                    [ij{1:2}] = ind2sub(dims, G.cells.indexMap(:));
                    ij        = [ij{:}];

                and likewise in 3D

                    [ijk{1:3}] = ind2sub(dims, G.cells.indexMap(:));
                    ijk        = [ijk{:}];

                In the latter case, ijk(i,:) is global (I,J,K) index of cell i.

            - G.cells.volumes (np.ndarray)
                Array with shape (G.cells.num,1) of cell volumes.

            - G.cells.centroids (np.ndarray)
                Array with shape (G.cells.num, d) of cell centroids in R^d.


        FACES: Face structure G.faces has the following attributes:

            - G.faces.num (int)
                Number of global faces in the grid.

            - G.faces.nodePos (np.ndarray)
                Array with shape (G.faces.num,1) into the `G.faces.faceNodes`
                array. Specifically, the node information of face `i` is found
                in the subarray

                    G.faces.nodes[G.faces.nodePos[i,0] : G.faces.nodePos[i+1,0]]

                The number of nodes for each face may be computed using the
                statement `np.diff(G.faces.nodePos, axis=0)`.

            - G.faces.nodes (np.ndarray)
                Array with shape (G.faces.nodePos[-1,0], 2) of vertices in the
                grid. Specifically, if `G.faces.nodes[i,0] == j` then local
                vertex `i` is part of global face number `j` and corresponds to
                global vertex `G.faces.nodes[i,1]`. To conserve memory, only
                the last column is stored. The first column can be constructed
                using the statement

                    # TODO: This is untested code
                    prst.utils.rldecode(
                        np.arange(G.faces.num, np.diff(G.faces.nodePos, axis=0))

            - G.faces.neighbors (np.ndarray)
                Array with shape (G.faces.num, 2) of neighboring information.
                Global face number `i` is shared by global cells
                `G.faces.neighbors[i,0]` and `G.faces.neighbors[i,1]`. One of
                `G.faces.neighbors[i,0]` and `G.faces.neighbors[i,1]` may be
                -1, but not both, meaning that face `i` is an external face
                shared only by a single cell (the nonnegative one).

            - G.faces.tag (np.ndarray)
                Array with shape (G.faces.num, 1) of face tags. A tag is a
                scalar. The exact semantics of this field is currently
                undecided and subject to change in future releases of PRST.

            - G.faces.areas (np.ndarray)
                Array with shape (G.faces.num, 1) of face areas.

            - G.faces.normals (np.ndarray)
                Array with shape (G.faces.num, d) of *AREA WEIGHTED*, directed
                face normals in R^d. The normal on face `i` points from cell
                `G.faces.neighbors[i,0]` to cell `G.faces.neighbors[i,1]`.

            - G.faces.centroids (np.ndarray)
                Array with shape (G.faces.num, d) of face centroids in R^d.


        NODES: Node structure G.nodes contains the following attributes:

            - G.nodes.num (int)
                Number of global nodes (vertices) in the grid.

            - G.nodes.coords (np.ndarray)
                Array with shape (G.nodes.num, d) of physical nodal coordinates
                in R^d. Global node `i` is at the physical coordinate
                `G.nodes.coords[i,:]`.

    Remarks:
        The grid is constructed according to a right-handed coordinate system
        where the z-coordinate is interpreted as depth. Consequently, plotting
        routines such as plotGrid display the grid with a reversed z-axis.
    """
    class Cells(object):
        def __str__(self):
            return str(self.__dict__.keys())

    class Faces(object):
        def __str__(self):
            return str(self.__dict__.keys())

    class Nodes(object):
        def __str__(self):
            return str(self.__dict__.keys())

    def __init__(self):
        self.gridType = []
        self.cells = self.Cells()
        self.faces = self.Faces()
        self.nodes = self.Nodes()

    def __eq__(G, V):
        """Tests two grids for equality.

        The grids are considered equal if the following attributes are equal
        for both grids:

            * cells.num
            * cells.facePos
            * cells.indexMap
            * cells.faces
            * faces.num
            * faces.nodePos
            * faces.neighbors
            * faces.nodes
            * nodes.num
            * nodes.coords (approximate floating point comparions)
            * gridType
            * gridDim
            * cartDims

        Example:

            >>> G = cartGrid(np.array([4, 5]))
            >>> V = cartGrid(np.array([4, 6]))
            >>> G == G
            True
            >>> G == V
            False

        """
        if ((G.cells.num != V.cells.num) or
           (not np.array_equal(G.cells.facePos, V.cells.facePos)) or
           (
               hasattr(G.cells, "indexMap") and hasattr(V.cells, "indexMap") and
               not np.array_equal(G.cells.indexMap, V.cells.indexMap)
           ) or
           (not np.array_equal(G.cells.faces, V.cells.faces)) or
           (G.faces.num != V.faces.num) or
           (not np.array_equal(G.faces.nodePos, V.faces.nodePos)) or
           (not np.array_equal(G.faces.neighbors, V.faces.neighbors)) or
           (not np.array_equal(G.faces.nodes, V.faces.nodes)) or
           (G.nodes.num != V.nodes.num) or
           (
               np.array_equal(G.nodes.coords.shape, V.nodes.coords.shape) and
               not np.isclose(G.nodes.coords, V.nodes.coords).all()
           ) or
           (G.gridType != V.gridType) or
           (G.gridDim != V.gridDim)):
            return False

        if hasattr(G.cells, "indexMap") ^ hasattr(V.cells, "indexMap"):
            return False

        if hasattr(G, "cartDims") and hasattr(V, "cartDims"):
            if not np.array_equal(G.cartDims, V.cartDims):
                return False
        elif hasattr(G, "cartDims") or hasattr(V, "cartDims"):
            return False

        # computeGeometry-attributes are compared approximately
        for obj1, obj2, attr in [
                (G.cells, V.cells, "volumes"),
                (G.cells, V.cells, "centroids"),
                (G.faces, V.faces, "areas"),
                (G.faces, V.faces, "normals"),
                (G.faces, V.faces, "centroids")]:
            if hasattr(obj1, attr) ^ hasattr(obj2, attr):
                # One object has the attribute while the other doesn't
                return False
            if hasattr(obj1, attr) and hasattr(obj2, attr):
                # Both objects have the attribute -- compare it
                attr1, attr2 = getattr(obj1, attr), getattr(obj2, attr)
                try:
                    if not np.isclose(attr1, attr2).all():
                        return False
                except Exception as e:
                    # TODO: Remove print debug
                    print(e)
                    return False

        return True

    def __ne__(G, V):
        return not G == V

    def _cmp(G, V):
        """Shows attributes comparions betwen two grids. For debugging."""
        prst.utils.recursive_diff(G, V)


    def __str__(G):
        s = "<PRST grid"
        for key, val in six.iteritems(G.__dict__):
            s += "\n  {}: {}".format(key, val)
        s += "\n>"
        return s

    def computeFaceNodes(G, aslist=False):
        """Compute array of face nodes.

        Synopsis:
            faceNodes = G.computeFaceNodes()

        Arguments:
            aslist (bool):
                If True, return list of lists, instead of list of ndarrays.

        Returns: *PYTHON LIST* of ndarrays. Each ndarray contains node indices
        for a face.

        Using the G.faces.nodePos attribute, calculate an explicit list of
        nodes for each face. For example, the grid given by `G =
        cartGrid([2,2,1])`.

        has the G.faces.nodePos array

            [[0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80]].T

        `G.computeFaceNodes()` will then return the Python list (!) of ndarrays

             [[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11],
              ...
              [68, 69, 70, 71],
              [72, 73, 74, 75],
              [76, 77, 78, 79]].

        This is useful since this is the format used by the MayaVi/VTK
        visualization libraries.
        """
        if aslist:
            return [range(a, b) for a,b in zip(G.faces.nodePos[:-1,0], G.faces.nodePos[1:,0])]
        else:
            return [np.arange(a, b) for a,b in zip(G.faces.nodePos[:-1,0], G.faces.nodePos[1:,0])]

##############################################################################
# GRID CONSTRUCTORS
##############################################################################

def tensorGrid(x, y, z=None, depthz=None):
    """Construct Cartesian grid with variable physical cell sizes.

    Synopsis:
        G = tensorGrid(x, y)
        G = tensorGrid(x, y, depthz=dz)
        G = tensorGrid(x, y, z)
        G = tensorGrid(x, y, z, depthz=dz)

    Args:
        x, y, z (ndarray): Vectors giving cell vertices, in units of meters, of
            individual coordinate directions. Specifically, the grid cell at
            logical location (i,j,k) will have a physical dimension of
            [x[i+1]-x[i],  y[j+1]-y[j],  z[k+1]-z[k]] meters.

        depthz (Optional[ndarray]): Depth, in units of meters, at which upper reservoir nodes
            are encountered. Assumed to be a len(x)-by-len(y) array of nodal
            depths. Default is np.zero([len(x), len(y)]) (i.e., the top of
            reservoir at zero depth).

    Returns:
        G - Grid structure mostly as detailed in grid_structure, though lacking the fields
            - G.cells.volumes
            - G.cells.centroids
            - G.faces.areas
            - G.faces.normals
            - G.faces.centroids

        These fields may be comptued using the function computeGeometry.

        There is, however, an additional field not described in grid_structure:
            - cartDims -- A length 2 or 3 vector giving number of cells in each
                          coordinate direction. In other words

                                cartDims == cellDim .

        G.cells.faces[:,1] contains integers 0-5 corresponding to directions
        W, E, S, N, T, B respectively.

    See also:
        grid_structure, computeGeometry

    """
    if z is None:
        G = _tensorGrid2D(x, y, depthz=depthz)
    else:
        G = _tensorGrid3D(x, y, z, depthz=depthz)

    # Record grid constructor in grid.
    G.gridType.append("tensorGrid")

    # Record number of dimensions
    G.gridDim = len(G.cartDims)

    return G


def _tensorGrid2D(x, y, depthz=None):
    # Check input data
    dx = np.diff(x)
    if any(dx <= 0):
        raise ValueError("x-values not monotonously increasing")
    dy = np.diff(y)
    if any(dy <= 0):
        raise ValueError("y-values not monotonously increasing")

    # We want float64 coordinates
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    sizeX, sizeY = len(x) - 1, len(y) - 1
    cellDim = np.array([sizeX, sizeY])

    numCells = sizeX * sizeY
    numNodes = (sizeX+1) * (sizeY+1)
    numFacesX = (sizeX+1) * sizeY # Faces parallel to y-axis
    numFacesY = sizeX * (sizeY+1) # Faces parallel to x-axis
    numFaces = numFacesX + numFacesY

    # Nodes
    nodesX, nodesY = np.meshgrid(x, y, indexing="ij")
    coords = np.column_stack((
        nodesX.ravel(order="F"), nodesY.ravel(order="F")
    ))

    ## Generate face-edges

    # Node index matrix
    nodeIndices = np.reshape(np.arange(0, numNodes),
            (sizeX+1, sizeY+1), order="F")

    # x-faces
    nodesFace1 = nodeIndices[:, :-1].ravel(order="F")
    nodesFace2 = nodeIndices[:, 1:].ravel(order="F")
    # Interleave the two arrays
    faceNodesX = np.zeros((len(nodesFace1) + len(nodesFace2)), dtype=np.int32)
    faceNodesX[0::2] = nodesFace1
    faceNodesX[1::2] = nodesFace2

    # y-faces
    nodesFace1 = nodeIndices[:-1, :].ravel(order="F")
    nodesFace2 = nodeIndices[1:, :].ravel(order="F")
    # Interleave the two arrays
    faceNodesY = np.zeros((len(nodesFace1) + len(nodesFace2)), dtype=np.int32)
    faceNodesY[0::2] = nodesFace2
    faceNodesY[1::2] = nodesFace1
    # Note: Nodes need to be reversed to obtain normals pointing in positive
    # i-direction in computeGeometry

    ## Assemble grid_structure faceNodes structure
    faceNodes = np.concatenate((faceNodesX, faceNodesY))

    ## Generate cell faces
    faceOffset = 0
    # Face index matrices
    facesX = np.reshape(faceOffset + np.arange(0, numFacesX),
            (sizeX+1, sizeY), order="F")
    faceOffset += numFacesX
    facesY = np.reshape(faceOffset + np.arange(0, numFacesY),
            (sizeX, sizeY+1), order="F")

    facesWest = facesX[:-1, :].ravel(order="F")   # west = 0
    facesEast = facesX[1:, :].ravel(order="F")   # east = 1
    facesSouth = facesY[:, :-1].ravel(order="F") # south = 2
    facesNorth = facesY[:, 1:].ravel(order="F")  # north = 3

    assert len(facesNorth) == len(facesEast) == len(facesSouth) == len(facesWest)
    cellFaces = np.zeros((4*len(facesWest), 2), dtype=np.int32)
    cellFaces[:,0] = np.column_stack(
            (facesWest, facesSouth, facesEast, facesNorth)
        ).ravel()
    cellFaces[:,1] = np.tile(np.array([0, 2, 1, 3]), len(facesWest))

    ## Generate neighbors

    # Cell index matrix
    cellIndices = -np.ones((sizeX+2, sizeY+2))
    cellIndices[1:-1, 1:-1] = np.arange(0, numCells)\
        .reshape((sizeX, sizeY), order="F")

    neighborsX1 = cellIndices[:-1, 1:-1].ravel(order="F")
    neighborsX2 = cellIndices[1:, 1:-1].ravel(order="F")
    neighborsY1 = cellIndices[1:-1, :-1].ravel(order="F")
    neighborsY2 = cellIndices[1:-1, 1:].ravel(order="F")

    neighbors = np.column_stack([
            np.concatenate([neighborsX1, neighborsY1]),
            np.concatenate([neighborsX2, neighborsY2])
        ])

    G = Grid()
    G.cells.num = numCells
    G.cells.facePos = np.arange(0, (numCells+1)*4-1, 4, dtype=np.int32)[:,np.newaxis]
    G.cells.indexMap = np.arange(0, numCells, dtype=np.int32)[:,np.newaxis]
    G.cells.faces = cellFaces.astype(np.int32)
    G.faces.num = numFaces
    G.faces.nodePos = np.arange(0, (numFaces+1)*2-1, 2)[:,np.newaxis]
    G.faces.neighbors = neighbors.astype(np.int32)
    G.faces.tag = np.zeros((numFaces, 1))
    G.faces.nodes = faceNodes.astype(np.int32)[:,np.newaxis]
    G.nodes.num = numNodes
    G.nodes.coords = coords
    G.cartDims = cellDim.astype(np.int32)

    return G


def _tensorGrid3D(x, y, z, depthz=None):
    # Check input data
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    if any(dx <= 0):
        raise ValueError("x-values not monotonously increasing")
    if any(dy <= 0):
        raise ValueError("y-values not monotonously increasing")
    if any(dz <= 0):
        raise ValueError("z-values not monotonously increasing")

    # We want float64 coordinates
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    z = z.astype(np.float64)

    sizeX, sizeY, sizeZ = len(x)-1, len(y)-1, len(z)-1
    cellDim = np.array([sizeX, sizeY, sizeZ])
    if depthz is None:
        depthz = np.zeros([sizeX+1, sizeY+1], dtype=np.float64)
    elif depthz.size != (sizeX+1) * (sizeY+1):
        raise ValueError("Argument depthz is wrongly sized")

    numCells = sizeX * sizeY * sizeZ
    numNodes = (sizeX+1) * (sizeY+1) * (sizeZ+1)
    numFacesX = (sizeX+1) * sizeY * sizeZ # Number of faces parallel to yz-plane
    numFacesY = sizeX * (sizeY+1) * sizeZ # Nubmer of faces parallel to xz-plane
    numFacesZ = sizeX * sizeY * (sizeZ+1) # Number of faces parallel to xy-plane
    numFaces = numFacesX + numFacesY + numFacesZ

    # Nodes/coordinates
    nodesX, nodesY, nodesZ = np.meshgrid(x, y, z, indexing="ij")
    for k in range(0, nodesZ.shape[2]):
        nodesZ[:,:,k] += depthz

    coords = np.column_stack([
            nodesX.ravel(order="F"),
            nodesY.ravel(order="F"),
            nodesZ.ravel(order="F"),
        ])

    ## Generate face-edges

    # Node index matrix
    nodeIndices = np.reshape(np.arange(0, numNodes),
            (sizeX+1,  sizeY+1, sizeZ+1), order="F")

    # X-faces
    nodesFace1 = nodeIndices[:, :-1, :-1].ravel(order="F")
    nodesFace2 = nodeIndices[:, 1:, :-1].ravel(order="F")
    nodesFace3 = nodeIndices[:, 1:, 1:].ravel(order="F")
    nodesFace4 = nodeIndices[:, :-1, 1:].ravel(order="F")
    faceNodesX = np.row_stack([nodesFace1, nodesFace2, nodesFace3, nodesFace4]).ravel(order="F")

    # Y-faces
    nodesFace1 = nodeIndices[:-1, :, :-1].ravel(order="F")
    nodesFace2 = nodeIndices[:-1, :, 1:].ravel(order="F")
    nodesFace3 = nodeIndices[1:, :, 1:].ravel(order="F")
    nodesFace4 = nodeIndices[1:, :, :-1].ravel(order="F")
    faceNodesY = np.row_stack([nodesFace1, nodesFace2, nodesFace3, nodesFace4]).ravel(order="F")

    # Z-faces
    nodesFace1 = nodeIndices[:-1, :-1, :].ravel(order="F")
    nodesFace2 = nodeIndices[1:, :-1, :].ravel(order="F")
    nodesFace3 = nodeIndices[1:, 1:, :].ravel(order="F")
    nodesFace4 = nodeIndices[:-1, 1:, :].ravel(order="F")
    faceNodesZ = np.row_stack([nodesFace1, nodesFace2, nodesFace3, nodesFace4]).ravel(order="F")

    faceNodes = np.concatenate([faceNodesX, faceNodesY, faceNodesZ])

    ## Generate cell faces
    faceOffset = 0
    # Face index matrices
    facesX = (np.arange(0, numFacesX) + faceOffset).reshape(
                (sizeX+1, sizeY, sizeZ), order="F")
    faceOffset += numFacesX
    facesY = (np.arange(0, numFacesY) + faceOffset).reshape(
                (sizeX, sizeY+1, sizeZ), order="F")
    faceOffset += numFacesY
    facesZ = (np.arange(0, numFacesZ) + faceOffset).reshape(
                (sizeX, sizeY, sizeZ+1), order="F")

    facesWest = facesX[:-1,:,:].ravel(order="F") # 0
    facesEast = facesX[1:,:,:].ravel(order="F")  # 1
    facesSouth = facesY[:,:-1,:].ravel(order="F")# 2
    facesNorth = facesY[:,1:,:].ravel(order="F") # 3
    facesTop = facesZ[:,:,:-1].ravel(order="F")  # 4
    facesBottom = facesZ[:,:,1:].ravel(order="F")# 5

    assert len(facesWest) == len(facesEast) == len(facesSouth) == len(facesNorth)
    assert len(facesNorth) == len(facesTop) == len(facesBottom)
    NUM_DIRECTIONS = 6
    cellFaces = np.zeros([NUM_DIRECTIONS*len(facesWest), 2], dtype=np.int32)
    cellFaces[:,0] = np.column_stack([
            facesWest, facesEast, facesSouth, facesNorth, facesTop, facesBottom
        ]).ravel()
    cellFaces[:,1] = np.tile(np.array([0, 1, 2, 3, 4, 5]), len(facesWest))

    ## Generate neighbors

    # Cell index matrix
    cellIndices = -np.ones([sizeX+2, sizeY+2, sizeZ+2])
    cellIndices[1:-1, 1:-1, 1:-1] = np.arange(0, numCells).reshape(
        [sizeX, sizeY, sizeZ], order="F")

    neighborsX1 = cellIndices[:-1, 1:-1, 1:-1].ravel(order="F")
    neighborsX2 = cellIndices[1:, 1:-1, 1:-1].ravel(order="F")
    neighborsY1 = cellIndices[1:-1, :-1, 1:-1].ravel(order="F")
    neighborsY2 = cellIndices[1:-1, 1:, 1:-1].ravel(order="F")
    neighborsZ1 = cellIndices[1:-1, 1:-1, :-1].ravel(order="F")
    neighborsZ2 = cellIndices[1:-1, 1:-1, 1:].ravel(order="F")
    neighbors = np.column_stack([
            np.concatenate([neighborsX1, neighborsY1, neighborsZ1]),
            np.concatenate([neighborsX2, neighborsY2, neighborsZ2]),
        ])

    ## Generate cell nodes
    ## NOT IMPLEMENTED IN PYRST

    ## Assemble grid object
    G = Grid()
    G.cells.num = numCells
    G.cells.facePos = np.arange(0, (numCells+1)*6-1, 6).astype(np.int32)
    G.cells.facePos = G.cells.facePos[:,np.newaxis] # As a column array
    G.cells.indexMap = np.arange(0, numCells).astype(np.int32)[:,np.newaxis]
    G.cells.faces = cellFaces.astype(np.int32)
    G.faces.num = numFaces
    G.faces.nodePos = np.arange(0, (numFaces+1)*4-1, 4).astype(np.int32)[:,np.newaxis]
    G.faces.neighbors = neighbors.astype(np.int32)
    G.faces.tag = np.zeros((numFaces,1))
    G.faces.nodes = faceNodes.astype(np.int32)[:,np.newaxis]
    G.nodes.num = numNodes
    G.nodes.coords = coords
    G.cartDims = cellDim.astype(np.int32)

    return G

def cartGrid(cellDim, physDim=None):
    """Constructs 2D or 3D Cartesian grid in physical space.

    Synopsis:
        G = cartGrid(cellDim)
        G = cartGrid(cellDim, physDim)

    Args:
        cellDim (ndarray or list):
            Specifies number of cells in each coordinate direction. Length must
            be 2 or 3.

        physDim (Optional[ndarray or list]):
            Specifies physical size of computational domain in meters.  Length
            must be same as celldim. Default value == celldim (i.e., each cell
            has physical dimension 1-by-1-by-1 meter).

    Returns:
        G - Grid structure mostly as detailed in grid_structure, though lacking
            the fields
                - G.cells.volumes
                - G.cells.centroids
                - G.faces.areas
                - G.faces.normals
                - G.faces.centroids

            These fields may be computed using the function computeGeometry.

            There is, however, an additional field not described in grid_structure:
                - cartDims -- A length 2 or 3 vector giving number of cells in each
                              coordinate direction. In other words

                                    cartDims == celldim .

            G.cells.faces[:,1] contains integers 0-5 corresponding to directions
            W, E, S, N, T, B respectively.

    Example:
        # Make a 10-by-5-2 grid on the unit cube.
        nx, ny, nz = 10, 5, 2
        G = cartGrid([nx, ny, nz], [1, 1, 1]);

        # Plot the grid in 3D-view.
        fig = plotGrid(G);

    See also:

        grid_structure, tensorGrid, computeGeometry

    """
    if not isinstance(cellDim, np.ndarray):
        cellDim = np.array(cellDim)
    assert isinstance(cellDim, np.ndarray)

    if any(cellDim <= 0):
        raise ValueError("cellDim must be positive")

    if physDim is None:
        physDim = cellDim
    if not isinstance(physDim, np.ndarray):
        physDim = np.array(physDim)
    assert isinstance(physDim, np.ndarray)

    if len(cellDim) == 3:
        x = np.linspace(0, physDim[0], cellDim[0]+1)
        y = np.linspace(0, physDim[1], cellDim[1]+1)
        z = np.linspace(0, physDim[2], cellDim[2]+1)
        G = tensorGrid(x, y, z)
    elif len(cellDim) == 2:
        x = np.linspace(0, physDim[0], cellDim[0]+1)
        y = np.linspace(0, physDim[1], cellDim[1]+1)
        G = tensorGrid(x, y)
    else:
        raise ValueError("Only 2- and 3-dimensional grids are supported.")

    # Record grid constructor in grid
    G.gridType.append("cartGrid")

    return G

def computeGeometry(G, findNeighbors=False, hingenodes=None):
    """
    Compute and add geometry attributes to grid object.

    Synopsis:
        G = computeGeometry(G)

    Arguments:
        G (Grid): Grid object. Input argument is mutated.

        findNeighbors (Optional[bool]):
            Force finding the neighbors array even if it exists. Defaults to
            False.


    Returns:
        G - Grid object with added attributes:
            - cells
                - volumes   -- An array with size G.cells.num of cell volumes.

                - centroids -- An array with shape (G.cells.num, G.gridDim)
                               of approximate cell centroids.

            - faces
                - areas     -- An array with size G.faces.num of face areas.

                - normals   -- An array with shape (G.faces.num, G.gridDim)
                               of face normals

                - centroids -- An array with shape (G.faces.num, G.griddim) of
                               approximate face centroids.

    Individual face normals have length (i.e. Euclidean norm) equal to the
    corresponding face areas. In other words, subject to numerical round-off,
    the identity

        np.linalg.norm(G.faces.normals[i,:], 2) == G.faces.areas[i]

    holds for all faces i in range(0, G.faces.num) .

    In three space dimensions, i.e. when G.gridDim == 3, the function
    `computeGeometry` assumes that the nodes on a given face `f` are ordered
    such that the face normal on `f` is directed from cell
    `G.faces.neighbors[f,0]` to cell `G.faces.neighbors[f,1]`.

    """

    ## Setup
    assert hingenodes is None or G.gridDim == 3,\
           "Hinge nodes are only supported for 3D grids"

    numCells = G.cells.num
    numFaces = G.faces.num

    ## Possibly find neighbors
    if findNeighbors:
        G.faces.neighbors = _findNeighbors(G)
        G = _findNormalDirections(G)
    else:
        if not hasattr(G.faces, "neighbors"):
            import prst
            prst.log.warn("No field faces.neighbors found. "
                   + "Adding plausible values... proceed with caution!")
            G.faces.neighbors = _findNeighbors(G)
            G = _findNormalDirections(G)

    ## Main part
    if G.gridDim == 3:
        ## 3D grid
        assert G.nodes.coords.shape[1] == 3
        faceNumbers = utils.rldecode(np.arange(G.faces.num), np.diff(G.faces.nodePos,axis=0))
        nodePos = G.faces.nodePos;
        nextNode = np.arange(1, G.faces.nodes.size+1)
        nextNode[nodePos[1:,0]-1] = nodePos[:-1,0]

        # Divide each face into sub-triangles all having one node as pCenter =
        # sum(nodes) / numNodes. Compute area-weighted normals, and add to
        # obtain approx face-normals. Compute resulting areas and centroids.
        import prst
        prst.log.info("Computing normals, areas and centroids...")
        # Construct a sparse matrix with zeros and ones.
        localEdge2Face = scipy.sparse.csc_matrix((
                np.ones(G.faces.nodes.size),
                (np.arange(G.faces.nodes.size), faceNumbers)
            ))
        # Divide each row in the matrix product by the numbers in the final
        # array elementwise
        pCenters = (localEdge2Face.transpose().dot(
                G.nodes.coords[G.faces.nodes[:,0]]
            )[None,:] / np.diff(G.faces.nodePos,axis=0))[0]
        pCenters = localEdge2Face.dot(pCenters)

        if hingenodes:
            raise NotImplementedError("hingenodes are not yet supported in PRST")

        subNormals = np.cross(
                  G.nodes.coords[G.faces.nodes[nextNode,0],:]
              - G.nodes.coords[G.faces.nodes[:,0],:],
                  pCenters - G.nodes.coords[G.faces.nodes[:,0],:]) / 2

        subAreas = np.linalg.norm(subNormals, axis=1)

        subCentroids = (G.nodes.coords[G.faces.nodes[:,0],:]
            + G.nodes.coords[G.faces.nodes[nextNode,0],:] + pCenters) / 3

        faceNormals = localEdge2Face.transpose().dot(subNormals)

        faceAreas = localEdge2Face.transpose().dot(subAreas)

        subNormalSigns = np.sign(np.sum(
            subNormals * (localEdge2Face * faceNormals), axis=1))

        faceCentroids = localEdge2Face.transpose().dot(
            subAreas[:,np.newaxis] * subCentroids
        ) / faceAreas[:,np.newaxis]

        # Computation above does not make sense for faces with zero area
        zeroAreaIndices = np.where(faceAreas <= 0)
        if np.any(zeroAreaIndices):
            import prst
            prst.log.warning("Faces with zero area detected. Such faces should be"
                      + "removed before calling computeGeometry")

        # Divide each cell into sub-tetrahedra according to sub-triangles above,
        # all having one node as cCenter = sum(faceCentroids) / #faceCentroids
        import prst
        prst.log.info("Computing cell volumes and centroids")
        cellVolumes = np.zeros(numCells)
        cellCentroids = np.zeros([numCells, 3])

        lastIndex = 0
        for cell in range(numCells):
            # Number of faces for the current cell
            cellNumFaces = G.cells.facePos[cell+1] - G.cells.facePos[cell]
            indexes = np.arange(cellNumFaces) + lastIndex

            # The indices to the faces for the current cell
            cellFaces = G.cells.faces[indexes,0]
            # triE are the row indices of the non-zero elements in the matrix, while
            # triF are the col indices of the same elements.
            # Original code based on MRST:
            #
            #   tmp = localEdge2Face[:,cellFaces]
            #   triEa, triFa = tmp.nonzero()
            #
            # Was very slow, so a custom function for extracting columns for a
            # csc_matrix was created. It is not fully tested, but is 4x faster.
            triF, triE = _csc_columns_nonzero(localEdge2Face.indptr,
                                              localEdge2Face.indices, cellFaces)


            cellFaceCentroids = faceCentroids[cellFaces,:]
            cellCenter = np.sum(cellFaceCentroids, axis=0) / cellNumFaces
            relSubC = subCentroids[triE,:] - cellCenter[np.newaxis,:]

            # The normal of a face f is directed from cell G.faces.neighbors[f,0]
            # to cell G.faces.neighbors[f,1].  If cell c is in the second column
            # for face f, then the normal must be multiplied by -1 to be an outer
            # normal.
            orientation = 2 * (G.faces.neighbors[G.cells.faces[indexes,0], 0] == cell) - 1

            outNormals = subNormals[triE,:] * (
                (subNormalSigns[triE] * orientation[triF])[:,np.newaxis]
            )
            tetraVolumes = (1/3) * np.sum(relSubC * outNormals, axis=1)
            tetraCentroids = (3/4) * relSubC;

            cellVolume = np.sum(tetraVolumes)
            # Warning: This expression can be very close to zero, and often differs from
            # the same calculation in MATLAB.
            relCentroid = (tetraVolumes.dot(tetraCentroids)) / cellVolume
            cellCentroid = relCentroid + cellCenter
            cellVolumes[cell] = cellVolume
            cellCentroids[cell,:] = cellCentroid

            lastIndex += cellNumFaces

    elif G.gridDim == 2 and G.nodes.coords.shape[1] == 2:
        # Sometimes G.cells.faces has a second column with face directions.
        # So we retrieve the index column only.
        cellFaces = G.cells.faces[:,0]
        ## 2D grid in 2D space
        import prst
        prst.log.info("Computing normals, areas and centroids")
        edges = G.faces.nodes.reshape([-1,2], order="C")
        # Distance between edge nodes as a vector. "Length" is misleading.
        edgeLength =   G.nodes.coords[edges[:,1],:] \
                     - G.nodes.coords[edges[:,0],:]

        # Since this is 2D, these are actually lengths
        faceAreas = np.linalg.norm(edgeLength, axis=1)
        faceCentroids = np.average(G.nodes.coords[edges], axis=1)
        faceNormals = np.column_stack([edgeLength[:,1], -edgeLength[:,0]])
        import prst
        prst.log.info("Computing cell volumes and centroids")
        numFaces = np.diff(G.cells.facePos, axis=0)[:,0]
        cellNumbers = prst.utils.rldecode(np.arange(G.cells.num), numFaces)
        cellEdges = edges[cellFaces,:]
        r = G.faces.neighbors[cellFaces, 1] == cellNumbers
        # swap the two columns
        cellEdges[r, 0], cellEdges[r, 1] = cellEdges[r, 1], cellEdges[r, 0]

        cCenter = np.zeros([G.cells.num, 2])

        # npg.aggregate is similar to accumarray in MATLAB
        cCenter[:,0] = aggregate(cellNumbers,
                faceCentroids[cellFaces, 0]) / numFaces

        cCenter[:,1] = aggregate(cellNumbers,
                faceCentroids[cellFaces, 1]) / numFaces

        a = G.nodes.coords[cellEdges[:,0],:] - cCenter[cellNumbers,:]
        b = G.nodes.coords[cellEdges[:,1],:] - cCenter[cellNumbers,:]
        subArea = 0.5 * (a[:,0]*b[:,1] - a[:,1] * b[:,0])

        subCentroid = (  cCenter[cellNumbers,:]
                       + 2*faceCentroids[cellFaces,:])/3
        cellVolumes = aggregate(cellNumbers, subArea)

        cellCentroids = np.zeros([G.cells.num, 2], dtype=np.float64)

        cellCentroids[:,0] = aggregate(
                cellNumbers, subArea * subCentroid[:,0]) / cellVolumes

        cellCentroids[:,1] = aggregate(
                cellNumbers, subArea * subCentroid[:,1]) / cellVolumes

    elif G.gridDim == 2 and G.nodes.coords.shape[1] == 3:
        ## 2D grid in 3D space
        raise NotImplementedError(
                "computeGeometry not yet implemented for surface grids")
    else:
        raise ValueError("gridDim or nodes.coords have invalid values")

    ## Update grid
    G.faces.areas = faceAreas[:,np.newaxis]
    G.faces.normals = faceNormals
    G.faces.centroids = faceCentroids
    G.cells.volumes = cellVolumes[:,np.newaxis]
    G.cells.centroids = cellCentroids

    if not hasattr(G, "gridType"):
        import prst
        prst.log.warning("Input grid has no type")
        G.gridType = []

    G.gridType.append("computeGeometry")

    return G

#@numba.njit
def _sort_by_first_column(arr):
    for index in range(1, arr.shape[0]):
        currentvalue = arr[index,0]
        currentother = arr[index,1]
        position = index

        while position>0 and arr[position-1,0]>currentvalue:
            arr[position,0], arr[position,1] = arr[position-1,0], arr[position-1,1]
            position = position-1

        arr[position,0] = currentvalue
        arr[position,1] = currentother

#@numba.njit
def _csc_columns_nonzero(indptr, indices, columns):
    indices_pos = 0
    rowcol_indices = np.empty((100, 2), dtype=np.int32)
    for col in xrange(len(columns)):
        i = columns[col]
        for n in range(indptr[i], indptr[i+1]):
            rowcol_indices[indices_pos, 0] = indices[n]
            rowcol_indices[indices_pos, 1] = col
            indices_pos += 1
    _sort_by_first_column(rowcol_indices[:indices_pos])
    return rowcol_indices[:indices_pos,1], rowcol_indices[:indices_pos,0] # col, row

def _findNeighbors(G):
    """Finds plausible values for the G.faces.neighbors array.

    Synopsis:
        G.faces.neighbors = _findNeighbors(G)

    Arguments:
        G (Grid): PRST Grid object

    """
    # Internal faces
    cellNumbers = prst.utils.rldecode(np.arange(0, G.cells.num), np.diff(G.cells.facePos, axis=0))
    # use mergesort to obtain same j array as in MRST
    # try/except to handle 1D and 2D arrays
    j = np.argsort(G.cells.faces[:,0], kind="mergesort")
    cellFaces = G.cells.faces[j,0]
    cellNumbers = cellNumbers[j]
    halfFaces = np.where(cellFaces[:-1] == cellFaces[1:])[0]
    N = -np.ones([G.faces.num, 2], dtype=np.int)
    N[cellFaces[halfFaces],0] = cellNumbers[halfFaces]
    N[cellFaces[halfFaces+1],1] = cellNumbers[halfFaces+1]

    # Boundary faces
    isBoundary = np.ones(cellNumbers.size, dtype=bool)
    isBoundary[halfFaces] = False
    isBoundary[halfFaces+1] = False
    N[cellFaces[isBoundary], 0] = cellNumbers[isBoundary]
    return N

def _findNormalDirections(G):
    """
    Detects neighborship based on normal directions.
    """
    if G.gridDim != 3 or G.nodes.coords.shape[1] != 3:
        raise ValueError("Detecting neighborship based on normal directions "
                        +"is only supported for 3D grids.")

    # Assume convex faces. Compute average of node coordinates.
    faceCenters = _averageCoordinates(np.diff(G.faces.nodePos, axis=0),
                G.nodes.coords[G.faces.nodes[:,0],:])[0]

    cellCenters, cellNumbers = _averageCoordinates(
        np.diff(G.cells.facePos, axis=0), faceCenters[G.cells.faces[:,0], :])

    # Compute triple product v1 x v2 . v3 of vectors
    # v1 = faceCenters - cellCenters
    # v2 = n1 - fc
    # v3 = n2 - n1
    # n1 and n2 being the first and second nodes of the face. Triple product
    # should be positive for half-faces with positive sign.

    nodes1 = G.nodes.coords[G.faces.nodes[:,0][G.faces.nodePos[:-1,0]    ], :]
    nodes2 = G.nodes.coords[G.faces.nodes[:,0][G.faces.nodePos[:-1,0] + 1], :]

    v1 = faceCenters[G.cells.faces[:,0], :] - cellCenters[cellNumbers]
    v2 = nodes1[G.cells.faces[:,0], :] - faceCenters[G.cells.faces[:,0], :]
    v3 = nodes2[G.cells.faces[:,0], :] - nodes1[G.cells.faces[:,0], :]

    a = np.sum(np.cross(v1, v2) * v3, axis=1)
    sgn = 2 * (G.faces.neighbors[G.cells.faces[:,0], 0] == cellNumbers) - 1

    i = aggregate(G.cells.faces[:,0], a * sgn) < 0
    G.faces.neighbors[i,0], G.faces.neighbors[i,1] = \
            G.faces.neighbors[i,1], G.faces.neighbors[i,0]

    return G

def _averageCoordinates(n, c):
    no = prst.utils.rldecode(np.arange(n.size), n)
    # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    c1 = scipy.sparse.csr_matrix(
            (np.ones(no.size), (no, np.arange(no.size)))
        )

    c2 = np.hstack((c, np.ones([c.shape[0], 1])))
    c3 = c1 * c2
    # Divide coordinate columns with final column, and remove the final column
    c4 = (c3[:, :-1] / c3[:, -1][:,np.newaxis])
    return c4, no
