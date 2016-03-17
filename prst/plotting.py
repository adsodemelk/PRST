from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import numpy as np


#### Utility functions for conversion to vtkUnstructuredGrid.
def _get_cell_faces(G, cell_idx):
    # MRST equivalent: G.cells.faces(facePos(i) : facePos(i+1)-1, :)
    return G.cells.faces[G.cells.facePos[cell_idx]:G.cells.facePos[cell_idx+1],:]

def _get_cell_faces_num(G, cell_idx):
    """Get number of faces a certain cell has."""
    return G.cells.facePos[cell_idx+1] - G.cells.facePos[cell_idx]

def _get_cell_nodes(G, cell_idx):
    """Get array of nodes for a cell."""
    # To get face nodes in MRST:
    # (The second column in cells_faces are cell indices)
    # >> G.faces.nodes(G.faces.nodePos(cell_idx) : G.faces.nodePos(cell_idx+1)-1, 1)
    cell_faces = G.cells.faces[G.cells.facePos[cell_idx,0]:G.cells.facePos[cell_idx+1,0],0]

    # The iterator returns nodes for every cell face.
    # The union of these nodes contain the unique nodes for the cell.
    # = face1_nodes U face2_nodes U ...
    return reduce(
        np.union1d,
        (G.faces.nodes[
            G.faces.nodePos[face_idx,0]:G.faces.nodePos[face_idx+1,0]
        ] for face_idx in cell_faces)
    )

def _get_cells_faces_num(G):
    """Get number of faces for all cells."""
    return np.diff(G.cells.facePos, axis=0)

def _get_face_nodes(G, face_idx):
    """Get nodes of a certain face."""
    return G.faces.nodes[G.faces.nodePos[face_idx,0] : G.faces.nodePos[face_idx+1,0]]

def _get_face_nodes_num(G, face_idx):
    """Get number of nodes of a certain face."""
    return G.faces.nodePos[face_idx+1,0] - G.faces.nodePos[face_idx,0]

def createVtkUnstructuredGrid(G):
    """
    Creates a tvtk.UnstructuredGrid object from a PRST grid.

    Synopsis:

        vtkGrid = createVtkUnstructuredGrid(G)

    This is currently only used for plotting purposes.
    """
    try:
        from tvtk.api import tvtk
        from mayavi import mlab
        from mayavi.sources.vtk_data_source import VTKDataSource
    except ImportError as e:
        prst.log.error("Couldn't import", e)
    if G.gridDim == 2:
        raise NotImplementedError("Only 3d for now")
    # Initialize grid object with point coordinates
    ug = tvtk.UnstructuredGrid(points=G.nodes.coords)

    # We are now going to add cells as VTK polyhedrons, one by one
    # using the insert_next_cell method of the `ug` object.
    # The first argument is the shape type. In this case we will use the most
    # general VTK type, the polyhedron, where every cell can have a variable
    # amount of faces.
    poly_type = tvtk.Polyhedron().cell_type

    # The second argument to insert_next_cell is a list in the following format:
    # (num_cell_faces, num_face0_pts, id1, id2, id3, num_face1_pts,id_1, id2,
    #  id3, ...).
    # First, then number of faces.
    # Then, for each face: The number of nodes for this face, and the nodes of the faces.

    # Loop through cells
    for cell_idx in range(G.cells.num):
        pt_ids = []
        pt_ids.append(_get_cell_faces_num(G, cell_idx))
        # Loop through faces for this cell
        for face_idx in np.nditer(_get_cell_faces(G, cell_idx)[:,0]):
                pt_ids.append(_get_face_nodes_num(G, face_idx))
                pt_ids += list(_get_face_nodes(G, face_idx))
        ug.insert_next_cell(poly_type, pt_ids)

    return ug

def plotGrid(G, cell_data=None, bgcolor=(0.5,0.5,0.5), size=(400,300),
             show_edges=True, mlab_figure=True, mlab_show=True,
             colorbar=True, colorbar_kwargs=None):
    """Plot grid in MayaVi.

    Synopsis:
        plotGrid(G)
        plotGrid(G, cell_data)

    Arguments:
        G:
            PRST grid object

        cell_data (Optional[ndarray]):
            Array of shape (G.cells.num,) containing one scalar value for each
            cell.

        bgcolor (Optional[3-tuple]):
            Background color of figure as a tuple of 3 float numbers. Default
            is grey (0.5, 0.5, 0.5).  Useful for creating figures with a white
            background.

        size (Optional[2-tuple]):
            Figure size. Default is (400,300) pixels.

        show_edges (Optional[bool]):
            Show cell edges. Default is True.

        mlab_figure (Optional[bool]):
            Whether a new figure is created. Default is True: A new figure is
            created when this function is called. If False, the grid is plotted
            in a previous figure. If a previous figure does not exist, one is
            created.

        mlab_show (Optional[bool]):
            Whether or not to call mlab.show() to display the figure.
            If mlab_show=False the figure can be modified after plotting.

        colorbar (Optional[bool]):
            Whether or not to show a colorbar.

        colorbar_kwargs (Optional[dict]):
            Keyword arguments passed on to mlab.colorbar(). For example, to
            orient the colorbar vertically, let
            colorbar_kwargs={'orientation':'vertical'}.
            Default is {} (no arguments).

    Returns: None

    If more advanced data transformation is necessary, it is recommended to
    create a custom plotting function utilizing
    prst.plotting.createVtkUnstructuredGrid manually. See the source code of
    this function.

    Technical note: VtkUnstructuredGrid does not support face data. Only cell
    data and point data. This function can only display cell data for now, but
    point data should be easy to implement. Data scaling (e.g., scaling z-axis)
    is not supported in MayaVi either, but is possible to mimic using a custom
    plotting function and the extent=(0,1,0,1,0,1) parameter of
    mlab.pipeline.surface.
    """
    try:
        from tvtk.api import tvtk
        from mayavi import mlab
        from mayavi.sources.vtk_data_source import VTKDataSource
    except ImportError as e:
        prst.log.error("Couldn't import", e)

    ug = createVtkUnstructuredGrid(G)
    if not cell_data is None:
        ug.cell_data.scalars = cell_data
        ug.cell_data.scalars.name = "Cell values"

    vtkSrc = VTKDataSource(data=ug)

    if mlab_figure:
        mlab.figure(bgcolor=bgcolor, size=size)
        if size != (400,300):
            prst.warning("Custom size has no effect for mlab_figure=False")
    dataset = mlab.pipeline.add_dataset(vtkSrc, name="PRST cell data")

    # Yellow surface with translucent black wireframe.
    # VTK takes care of surface extraction, unlike in MRST where this is done
    # manually. The downside is performance, since the whole grid is converted.
    # On the other hand, this makes it possible to use various VTK filters to
    # transform the data.
    if cell_data is None:
        # MRST yellow
        mlab.pipeline.surface(dataset, opacity=1., color=(1,1,0))
    else:
        # Display using default colormap
        mlab.pipeline.surface(dataset, opacity=1.)
    if show_edges:
        mlab.pipeline.surface(mlab.pipeline.extract_edges(vtkSrc), color=(0,0,0), opacity=0.3)

    if colorbar:
        if colorbar_kwargs is None:
            colorbar_kwargs = {}
        mlab.colorbar(**colorbar_kwargs)

    if mlab_show:
        mlab.show()

def plotCellData(G, cell_data):
    """See plotGrid."""
    return plotGrid(G, cell_data)
