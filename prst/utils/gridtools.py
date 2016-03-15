"""
GRIDTOOLS



Functions in MRST:
  checkGrid             - Undocumented utility function
  compareGrids          - Determine if two grid structures are the same.
  connectedCells        - Compute connected components of grid cell subsets.
  findEnclosingCell     - Find cells with closest centroid (in Euclidian norm) in a 2D grid.
  getCellNoFaces        - Get a list over all half faces, accounting for possible NNC
  getConnectivityMatrix - Derive global, undirected connectivity matrix from neighbourship relation.
  getNeighbourship      - Retrieve neighbourship relation ("graph") from grid
  gridAddHelpers        - Add helpers to existing grid structure for cleaner code structure.
  gridCellFaces         - Find faces corresponding to a set of cells
  gridCellNo            - Construct map from half-faces to cells or cell subset
  gridCellNodes         - Extract nodes per cell in a particular set of cells
  gridFaceNodes         - Find nodes corresponding to a set of faces
  gridLogicalIndices    - Given grid G and optional subset of cells, find logical indices.
  indirectionSub        - Index map of the type G.cells.facePos, G.faces.nodePos...
  makePlanarGrid        - Construct 2D surface grid from faces of 3D grid.
  neighboursByNodes     - Derive neighbourship from common node (vertex) relationship
  removeNodes           - Undocumented utility function
  sampleFromBox         - Sample from data on a uniform Cartesian grid that covers the bounding box
  sortedges             - SORTEDGES(E, POS) sorts the edges given by rows in E.
  sortGrid              - Permute nodes, faces and cells to sorted form
  transform3Dto2Dgrid   - Transforms a 3D grid into a 2D grid.
  translateGrid         - Move all grid coordinates according to particular translation
  triangulateFaces      - Split face f in grid G into subfaces.
  volumeByGaussGreens   - Compute cell volume by means of Gauss-Greens' formula

Functions in PRST:
  getNeighborship       - Retrieve neighborship relation ("graph") from grid
"""
__all__ = ["getNeighborship"]

import numpy as np

def getNeighborship(G, kind="Geometrical", incBdry=False, nargout=1):
    """
    Retrieve neighborship relation ("graph") from grid.

    Synopsis:
        N, isnnc = getNeighborship(G, kind)

    Arguments:
        G (Grid):
            PRST grid object.

        kind (str):
            What kind of neighborship relation to extract. String. The
            following options are supported:

                - "Geometrical"
                    Extract geometrical neighborship relations. The geometric
                    connections correspond to physical, geometric interfaces
                    are are the ones listed in `G.faces.neighbors`.

                - "Topological"
                    Extract topological neighborship relations. In addition to
                    the geometrical relations  of "Geometrical" these possibly
                    include non-neighboring connections resulting from
                    pinch-out processing or explicit NNC lists in an ECLIPSE
                    input deck.

                    Additional connections will only be defined if the grid `G`
                    contains an `nnc` sub-structure.

        incBdry (bool):
            Flag to indicate whether or not to include boundary connections. A
            boundary connection is a connection in which one of the connecting
            cells is the outside (i.e., cell zero). Boolean. Default: False (Do
            NOT include boundary connections.)

        nargout (int):
            Set to 2 to return both N and isnnc. Default: Return only N.

    Returns:
        N (ndarray[int]):
            Neighborshop relation. An (m, 2)-shaped array of cell indices that
            form the connections--geometrical or otherwise. This array has
            similar interpretation to the field `G.faces.neighbors`, but may
            contain additional connections if kind="Topological".

        isnnc (ndarray[bool]):
            An (m, 1)-shaped boolean array indicating whether or not the
            corresponding connection (row) of N is a geometrical connection
            (i.e., a geometric interface from the grid `G`).

            Specifically, isnnc[i,0] is True if N[i,:] comes from a
            non-neighboring (i.e., non-geometrical) connection.

    Note:
        If the neighborship relation is later to be used to compute the graph
        adjacency matrix using function `getConnectivityMatrix`, then `incBdry`
        must be False.

    See also:
        processGRDECL (MRST), processPINCH (MRST), getConnectivityMatrix (MRST)
    """
    # Geometric neighborship (default)
    N = G.faces.neighbors

    if not incBdry:
        # Exclude boundary connections
        N = N[np.all(N != -1, axis=1), :]

    if nargout >= 2:
        isnnc = np.zeros((N.shape[0], 1), dtype=bool)

    if kind=="Topological" and hasattr(G, "nnc") and hasattr(G.nnc, "cells"):
        assert G.nnc.cells.shape[1] == 2
        N = np.vstack((N, G.nnc.cells))

    if nargout >= 2:
        try:
            isnnc = np.vstack((isnnc, np.ones((G.nnc.cells.shape[0], 1), 1)))
        except AttributeError:
            pass
        return N, isnnc
    else:
        return N

def getCellNoFaces(G):
    """
    Get a list of all half faces, accounting for possible NNC.

    Synopsis:
        cellNo, cellFaces, isNNC = getCellNoFaces(G)

    Description:
        This utility function is used to produce a listing of all half faces in
        a grid along with the respective cells they belong to. While relatively
        trivial for most grids, this function specifically accounts for
        non-neighboring connections / NNC.

    Arguments:
        G (Grid):
            Grid structure with optional .nnc.cells attribute.

    Returns:
        cellNo (ndarray):
            Column array with shape (M,1) where M is the number of geometric
            half-faces + 2 * number of NNC, where each entry corresponds to
            cell index of that half face.

        cellFaces (ndarray):
            Column array with shape (M,1) with M as above, where each entry is
            the connection index. For the first entries, this is simply the
            face number. Otherwise, it is the entry of the NNC connection.

    See also:
        prst.utils.rldecode
    """
    import prst.utils as utils # circular import

    cellNo = utils.rldecode(np.arange(G.cells.num), np.diff(G.cells.facePos,axis=0))[:,np.newaxis]
    cellFaces = G.cells.faces[:,0][:,np.newaxis]

    # Mapping to show which entries in cellNo/cellFaces are resulting from
    # NNC and not actual geometric faces.
    isNNC = np.zeros((len(cellNo), 1), dtype=np.bool)

    # If NNC is present, we add these extra connections to cellNo
    # and cellFaces to allow consisten treatment.
    if hasattr(G, 'nnc') and hasattr(G.nnc, "cells"):
        prst.warning("getCellNoFaces is untested for grids with NNC. Compare results with MRST.")
        # Stack columns into a single column
        nnc_cells = np.r_[G.nnc.cells[:,0], G.nnc.cells[:,1]][:,np.newaxis]
        # NNC is located at the end after the regular faces
        nnc_faceno = G.faces.num + np.arange(G.nnc.cells.shape[0])[:,np.newaxis]
        cellNo = np.r_[cellNo, nnc_cells] # Stack as 2d column
        cellFaces = np.r_[cellFaces, nnc_faceno, nnc_faceno] # Stack as column
        # Added connections are NNC
        isNNC = np.r_[isNNC, np.ones((len(nnc_cells),1), dtype=np.bool)]
    return cellNo, cellFaces, isNNC
