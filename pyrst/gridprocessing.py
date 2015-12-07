from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

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

    def __repr__(self):
        return "({}D-Grid, {} cells, {} faces)"\
            .format(self.gridDim, self.cells.num, self.faces.num)

    def __eq__(G, V):
        if ((G.cells.num != V.cells.num) or
           (not np.array_equal(G.cells.facePos, V.cells.facePos)) or
           (not np.array_equal(G.cells.indexMap, V.cells.indexMap)) or
           (not np.array_equal(G.cells.faces, V.cells.faces)) or
           (G.faces.num != V.faces.num) or
           (not np.array_equal(G.faces.nodePos, V.faces.nodePos)) or
           (not np.array_equal(G.faces.neighbors, V.faces.neighbors)) or
           (not np.array_equal(G.faces.nodes, V.faces.nodes)) or
           (G.nodes.num != V.nodes.num) or
           (not np.array_equal(G.nodes.coords, V.nodes.coords)) or
           (G.gridType != V.gridType) or
           (G.gridDim != V.gridDim)):
            print(V.gridType)
            return False

        if hasattr(G, "cartDims") and hasattr(V, "cartDims"):
            if not np.array_equal(G.cartDims, V.cartDims):
                print("cartDims uenqual")
                return False
        elif hasattr(G, "cartDims") or hasattr(V, "cartDims"):
            print("cartdims elif")
            return False

        return True


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
    pass

def cartGrid(celldim, physdim=None):
    """Constructs 2D or 3D Cartesian grid in physical space.

    Synopsis:
        G = cartGrid(celldim)
        G = cartGrid(celldim, physdim)

    Args:
        celldim (ndarray):
            Specifies number of cells in each coordinate direction. Length must
            be 2 or 3.

        physdim (Optional[ndarray]):
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
    if any(celldim > 0):
        raise ValueError("celldim must be positive")

    if physdim is None:
        physdim = celldim

    x = np.linspace(0, physdim[0], celldim[0]+1)
    y = np.linspace(0, physdim[1], celldim[1]+1)
    if len(celldim) == 3:
        z = np.linspace(0, physdim[2], celldim[2]+1)
        G = tensorGrid(x, y, z)
    else:
        G = tensorGrid(x, y)

    # Record grid constructor in grid
    G.gridType.add(__file__)

    return G
