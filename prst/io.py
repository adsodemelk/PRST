from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

from scipy.io import loadmat
import numpy as np


import prst
import prst.gridprocessing

__all__ = ["loadMRSTGrid",]

def loadMRSTGrid(matfile, variablename="G"):
    """Loads MRST grid as PRST grid.

    The grid is saved in MATLAB using the command

        save('mygrid.mat', 'G', '-v7')

    where `G` is the name of the grid variable. All indices are converted to be
    zero-indexed. E.g., Cell 1 in MATLAB will be renamed Cell 0 in Python, and
    so on.

    Args:
        matfile (str): Path to saved grid .mat-file.

        variablename (Optional[str]): Name of grid variable in .mat-file.
            Defaults to "G". This must be specified if the grid is named
            something else than "G".

    Returns:
        G - PRST grid with zero indexing.

    See the source code of this function to see which variables are modified to
    be zero-indexed. Some important consequences of zero-indexing are:

        * The G.faces.neighbors array will no longer use 0 to indicate that
          there are no neighbors. Instead, -1 is used.

    """
    # Convert data to these types
    INT_DTYPE = np.int64
    FLOAT_DTYPE = np.float64

    data = loadmat(matfile, squeeze_me=True, struct_as_record=False)
    M = data[variablename] # MRST grid data, one-indexed

    G = prst.gridprocessing.Grid()
    G.cells.num = M.cells.num
    G.cells.facePos = M.cells.facePos.astype(INT_DTYPE) - 1
    G.cells.facePos.shape = (G.cells.facePos.size,1)
    G.cells.faces = M.cells.faces.astype(INT_DTYPE) - 1
    if G.cells.faces.ndim == 1:
        # Make into column array
        G.cells.faces = G.cells.faces[:,np.newaxis]

    # computeGeometry attributes may not exist
    try:
        G.cells.indexMap = M.cells.indexMap.astype(INT_DTYPE) - 1
        G.cells.indexMap = G.cells.indexMap[:,np.newaxis] # make into column
    except AttributeError:
        prst.log.info("Loaded grid has no cells.indexMap")
    try:
        G.cells.volumes = M.cells.volumes.astype(FLOAT_DTYPE)
        G.cells.volumes = G.cells.volumes[:,np.newaxis]
    except AttributeError:
        prst.log.info("Loaded grid has no cells.volumes")
    try:
        G.cells.centroids = M.cells.centroids.astype(FLOAT_DTYPE)
    except AttributeError:
        prst.log.info("Loaded grid has no cells.centroids")
    try:
        G.faces.areas = M.faces.areas.astype(FLOAT_DTYPE)
        G.faces.areas = G.faces.areas[:,np.newaxis]
    except AttributeError:
        prst.log.info("Loaded grid has no faces.areas")
    try:
        G.faces.centroids = M.faces.centroids.astype(FLOAT_DTYPE)
    except AttributeError:
        prst.log.info("Loaded grid has no faces.centroids")
    try:
        G.faces.normals = M.faces.normals.astype(FLOAT_DTYPE)
    except AttributeError:
        prst.log.info("Loaded grid has no faces.normals")

    G.faces.num = M.faces.num
    G.faces.nodePos = M.faces.nodePos.astype(INT_DTYPE) - 1
    G.faces.nodePos = G.faces.nodePos[:,np.newaxis] # 2d column
    try:
        G.faces.neighbors = M.faces.neighbors.astype(INT_DTYPE) - 1
    except AttributeError:
        prst.log.warn("Loaded grid has no faces.neighbors")
    G.faces.nodes = M.faces.nodes.astype(INT_DTYPE) - 1
    G.faces.nodes = G.faces.nodes[:,np.newaxis] # 2d column

    G.nodes.num = M.nodes.num
    G.nodes.coords = M.nodes.coords.astype(FLOAT_DTYPE)

    try:
        G.cartDims = M.cartDims.astype(INT_DTYPE)
    except AttributeError:
        print("Info: Loaded grid has no cartDims") # LOG
    # Matlab saves the gridType either as string or array of strings, depending
    # on the number of grid types. We use "gridType" since type is a Python
    # keyword.
    if isinstance(M.type, six.string_types):
        G.gridType = [M.type]
    elif isinstance(M.type, np.ndarray):
        # Convert to normal Python list for convenience
        G.gridType = list(M.type)
    else:
        raise ValueError("gridType has unknown type " + M.type.__class__.__name__)
    G.gridDim = M.griddim

    return G

def saveVtkUnstructuredGrid(vtkGrid, file_name):
    """Writes vtkGrid to file. (.vtu XML format).

    To create a vtkGrid from a PRST grid:

        from prst.plotting import createVtkUnstructuredGrid
        vtkGrid = createVtkUnstructuredGrid(G)
    """
    try:
        from tvtk.api import tvtk
    except:
        prst.log.error("Couldn't import tvtk")
        return


    w = tvtk.XMLUnstructuredGridWriter(input=vtkGrid, file_name=file_name)
    return w.write()

