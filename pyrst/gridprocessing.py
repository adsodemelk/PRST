from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

__all__ = ["Grid", "tensorGrid", "cartGrid", "computeGeometry", "findNeighbors"]

class Grid:
    """TODO: Copy "grid_structure" help from MRST here"""
    class Cells:
        pass

    class Faces:
        pass

    class Nodes:
        pass

    def __init__(self):
        self.gridType = set()
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
           (not np.array_equal(G.cells.indexMap, V.cells.indexMap)) or
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

        if hasattr(G, "cartDims") and hasattr(V, "cartDims"):
            if not np.array_equal(G.cartDims, V.cartDims):
                return False
        elif hasattr(G, "cartDims") or hasattr(V, "cartDims"):
            return False

        return True

    def __ne__(G, V):
        return not G == V

    def _cmp(G, V):
        """Shows attributes comparions betwen two grids. For debugging.

        Example:

            >>> G = cartGrid(np.array([4, 5]))
            >>> V = cartGrid(np.array([4, 6]))
            >>> G._cmp(V)
            Grid attributes comparison:
                cells.num are different
                cells.facePos are different
                cells.indexMap are different
            ...
        """
        print("Grid attributes comparison:")
        s = {True: "are equal", False: "are different"}

        print("    cells.num", s[G.cells.num == V.cells.num])
        print("    cells.facePos", s[np.array_equal(G.cells.facePos, V.cells.facePos)])
        print("    cells.indexMap", s[np.array_equal(G.cells.indexMap, V.cells.indexMap)])
        print("    cells.faces", s[np.array_equal(G.cells.faces, V.cells.faces)])

        print("    faces.num", s[G.faces.num != V.cells.num])
        print("    faces.nodePos", s[np.array_equal(G.faces.nodePos, V.faces.nodePos)])
        print("    faces.neighbors", s[np.array_equal(G.faces.neighbors, V.faces.neighbors)])
        print("    faces.nodes", s[np.array_equal(G.faces.nodes, V.faces.nodes)])

        print("    nodes.num", s[G.nodes.num == V.nodes.num])
        print("    nodes.coords",
                s[np.array_equal(G.nodes.coords.shape, V.nodes.coords.shape) and
                  np.isclose(G.nodes.coords, V.nodes.coords).all()])
        print("    gridType", s[G.gridType == V.gridType])
        print("    gridDim", s[G.gridDim == V.gridDim])

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
    G.gridType.add("tensorGrid")

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
    faceNodesX = np.zeros((len(nodesFace1) + len(nodesFace2)))
    faceNodesX[0::2] = nodesFace1
    faceNodesX[1::2] = nodesFace2

    # y-faces
    nodesFace1 = nodeIndices[:-1, :].ravel(order="F")
    nodesFace2 = nodeIndices[1:, :].ravel(order="F")
    # Interleave the two arrays
    faceNodesY = np.zeros((len(nodesFace1) + len(nodesFace2)))
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
    cellFaces = np.zeros((4*len(facesWest), 2))
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
    G.cells.facePos = np.arange(0, (numCells+1)*4-1, 4)
    G.cells.indexMap = np.arange(0, numCells)
    G.cells.faces = cellFaces
    G.faces.num = numFaces
    G.faces.nodePos = np.arange(0, (numFaces+1)*2-1, 2)
    G.faces.neighbors = neighbors
    G.faces.tag = np.zeros((numFaces, 1))
    G.faces.nodes = faceNodes
    G.nodes.num = numNodes
    G.nodes.coords = coords
    G.cartDims = cellDim

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
    cellFaces = np.zeros([NUM_DIRECTIONS*len(facesWest), 2])
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
    G.cells.facePos = np.arange(0, (numCells+1)*6-1, 6)
    G.cells.indexMap = np.arange(0, numCells)
    G.cells.faces = cellFaces
    G.faces.num = numFaces
    G.faces.nodePos = np.arange(0, (numFaces+1)*4-1, 4)
    G.faces.neighbors = neighbors
    G.faces.tag = np.zeros((numFaces,1))
    G.faces.nodes = faceNodes
    G.nodes.num = numNodes
    G.nodes.coords = coords
    G.cartDims = cellDim

    return G

def cartGrid(cellDim, physDim=None):
    """Constructs 2D or 3D Cartesian grid in physical space.

    Synopsis:
        G = cartGrid(cellDim)
        G = cartGrid(cellDim, physDim)

    Args:
        cellDim (ndarray):
            Specifies number of cells in each coordinate direction. Length must
            be 2 or 3.

        physDim (Optional[ndarray]):
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
    assert isinstance(cellDim, np.ndarray)

    if any(cellDim <= 0):
        raise ValueError("cellDim must be positive")

    if physDim is None:
        physDim = cellDim
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
    G.gridType.add("cartGrid")

    return G


def computeGeometry(G):
    pass


def findNeighbors(G):
    """Finds plausible values for the G.faces.neighbors array.

    Synopsis:
        G.faces.neighbors = findNeighbors(G)

    Arguments:
        G (Grid): PyRST Grid object

    """
    # Internal faces
    cellNumbers = rldecode(np.arange(0, G.cells.num), np.diff(G.cells.facePos))
    print(G.cells.faces[113], G.cells.faces[54])
    # use mergesort to obtain same j array as in MRST
    j = np.argsort(G.cells.faces, kind='mergesort')
    cellFaces = G.cells.faces[j]
    cellNumbers = cellNumbers[j]
    halfFaces = np.where(cellFaces[:-1] == cellFaces[1:])[0]
    N = -np.ones([G.faces.num, 2], dtype=np.int)
    N[cellFaces[halfFaces],0] = cellNumbers[halfFaces]
    N[cellFaces[halfFaces+1],1] = cellNumbers[halfFaces+1]

    # Boundary faces
    isBoundary = np.ones(cellNumbers.size, dtype=np.bool)
    isBoundary[halfFaces] = False
    isBoundary[halfFaces+1] = False
    N[cellFaces[isBoundary], 0] = cellNumbers[isBoundary]
    return N
