from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import six

from scipy.io import loadmat
import numpy as np

import pyrst.gridprocessing

__all__ = ["loadMRSTGrid",]

def loadMRSTGrid(matfile, variablename="G"):
    """Loads MRST grid as PyRST grid.

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
        G - PyRST grid with zero indexing.

    See the source code of this function to see which variables are modified to
    be zero-indexed. Some important consequences of zero-indexing are:

        * The G.faces.neighbors array will no longer use 0 to indicate that
          there are no neighbors. Instead, -1 is used.

    """
    # Convert data to these types
    INT_DTYPE = np.int32
    FLOAT_DTYPE = np.float64

    data = loadmat(matfile, squeeze_me=True, struct_as_record=False)
    M = data[variablename] # MRST grid data, one-indexed

    G = pyrst.gridprocessing.Grid()
    G.cells.num = M.cells.num
    G.cells.facePos = M.cells.facePos.astype(INT_DTYPE) - 1
    G.cells.faces = M.cells.faces.astype(INT_DTYPE) - 1
    try:
        G.cells.indexMap = M.cells.indexMap.astype(INT_DTYPE) - 1
    except AttributeError:
        print("Info: Loaded grid has no indexMap") # LOG

    G.faces.num = M.faces.num
    G.faces.nodePos = M.faces.nodePos.astype(INT_DTYPE) - 1
    G.faces.neighbors = M.faces.neighbors.astype(INT_DTYPE) - 1
    G.faces.nodes = M.faces.nodes.astype(INT_DTYPE) - 1

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
        G.gridType = set([M.type])
    elif isinstance(M.type, np.ndarray):
        G.gridType = set(M.type)
    else:
        raise ValueError("gridType has unknown type " + M.type.__class__.__name__)
    G.gridDim = M.griddim

    return G
