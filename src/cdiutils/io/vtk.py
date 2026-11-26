import numpy as np
import networkx as nx

# handling vtk case
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    IS_VTK_AVAILABLE = True

except ImportError:
    IS_VTK_AVAILABLE = False


class VtkImportError(ImportError):
    """Custom exception to handle Vtk import error."""

    def __init__(self, msg: str = None) -> None:
        _msg = "vtk package is not installed."
        if msg is not None:
            _msg += "\n" + msg
        super().__init__(_msg)


def load_vtk(file_path: str):
    """
    Load a vtk file.

    Args:
        file_path (_type_): the path to the file to open.

    Raises:
        VtkImportError: if vtk is not installed.

    Returns:
        the reader output
    """
    if not IS_VTK_AVAILABLE:
        raise VtkImportError

    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(file_path)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.Update()

    return reader.GetOutput()


def save_as_vti(
    output_path: str,
    voxel_size: tuple | list | np.ndarray,
    cxi_convention: bool = False,
    origin: tuple = (0, 0, 0),
    **np_arrays: dict[np.ndarray],
) -> None:
    """Save numpy arrays to .vti file."""
    if not IS_VTK_AVAILABLE:
        raise VtkImportError
    voxel_size = tuple(voxel_size)
    nb_arrays = len(np_arrays)

    if not nb_arrays:
        raise ValueError(
            "np_arrays is empty, please provide a dictionary of "
            "(fieldnames: np.ndarray) you want to save."
        )
    is_init = False
    for i, (key, array) in enumerate(np_arrays.items()):
        if not is_init:
            shape = array.shape
            if cxi_convention:
                # change order to match VTK (x, y, z) convention
                voxel_size = (voxel_size[2], voxel_size[1], voxel_size[0])
                shape = (shape[2], shape[1], shape[0])
            image_data = vtk.vtkImageData()
            image_data.SetOrigin(origin)
            image_data.SetSpacing(voxel_size)
            image_data.SetExtent(
                0, shape[0] - 1, 0, shape[1] - 1, 0, shape[2] - 1
            )
            point_data = image_data.GetPointData()
            is_init = True
        if cxi_convention:
            # natural ravelling row-major order is the one to use here,
            # because, it will be x first, then y, then z.
            vtk_array = numpy_to_vtk(array.ravel())
        else:  # convention XU (x, y z)
            # here, row-major order does not match our needs, we need to
            # transpose the array.
            vtk_array = numpy_to_vtk(array.transpose(2, 1, 0).ravel())
        point_data.AddArray(vtk_array)
        point_data.GetArray(i).SetName(key)
        point_data.Update()

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(image_data)
    writer.Write()


def save_as_vtp(
    output_path: str,
    q_direction: np.ndarray,
    length: float = 100.0,
    origin: tuple = (0, 0, 0),
) -> None:
    """Save a q-vector as a 3D arrow in a VTP file."""
    if not IS_VTK_AVAILABLE:
        raise VtkImportError

    q_direction = np.asarray(q_direction, dtype=float)
    norm = np.linalg.norm(q_direction)

    if norm == 0:
        raise ValueError("q_direction cannot be zero.")

    q_direction /= norm

    start = np.asarray(origin, dtype=float)
    end = start + length * q_direction

    # Create the arrow along the x-axis by default.
    arrow_source = vtk.vtkArrowSource()
    arrow_source.SetTipLength(0.25)
    arrow_source.SetTipRadius(0.08)
    arrow_source.SetShaftRadius(0.025)

    # Orient the arrow from the x-axis towards q_direction.
    transform = vtk.vtkTransform()
    transform.Translate(*start)

    x_axis = np.array([1.0, 0.0, 0.0])
    rotation_axis = np.cross(x_axis, q_direction)
    rotation_angle = np.degrees(
        np.arccos(np.clip(np.dot(x_axis, q_direction), -1.0, 1.0))
    )

    if np.linalg.norm(rotation_axis) > 1e-10:
        rotation_axis /= np.linalg.norm(rotation_axis)
        transform.RotateWXYZ(rotation_angle, *rotation_axis)
    elif np.dot(x_axis, q_direction) < 0:
        transform.RotateWXYZ(180, 0, 0, 1)

    transform.Scale(length, length, length)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(arrow_source.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    # Write the arrow geometry.
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputConnection(transform_filter.GetOutputPort())
    writer.Write()

    
def save_dislocation_graph_as_vtp(
    graph: nx.Graph,
    filename: str,
    radius_nm: float = 2.0,
    resolution: int = 1,
) -> None:
    """Export a dislocation graph as VTK PolyData (.vtp).
    
    Connected graph edges are combined into continuous VTK polylines.
    Closed loops, branches, and connected components are preserved.
    """
    try:
        import vtk
    except ImportError as exc:
        raise ImportError(
            "VTK is required to export dislocation graphs as VTP. "
            "Install it with 'pip install vtk'."
        ) from exc

    if radius_nm <= 0:
        raise ValueError("radius_nm must be positive.")
    
    if resolution < 1:
        raise ValueError("resolution must be at least 1.")
    
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot export an empty dislocation graph.")
        
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    vertices = vtk.vtkCellArray()

    node_to_point = {}
    nodes = list(graph.nodes)

    # Store physical coordinates in nm.
    for node_id in nodes:
        xyz = graph.nodes[node_id]["xyz_nm"]
        point_index = points.InsertNextPoint(
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
        )
        node_to_point[node_id] = point_index


    # Track graph edges that have already been included in a polyline.
    visited_edges = set()

    def edge_key(source, target):
        return frozenset((source, target))

    def add_polyline(path):
        """Add a connected path as one VTK polyline."""
        if len(path) < 2:
            return

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(path))

        for index, node in enumerate(path):
            polyline.GetPointIds().SetId(
                index, node_to_point[node]
            )

        lines.InsertNextCell(polyline)

    # Process each connected component independently.
    for component in nx.connected_components(graph):
        component_graph = graph.subgraph(component)

        # Endpoints and junctions have degree != 2.
        critical_nodes = {
            node
            for node, degree in component_graph.degree()
            if degree != 2
        }

        # Trace paths starting from endpoints and junctions.
        for start in critical_nodes:
            for neighbour in component_graph.neighbors(start):
                key = edge_key(start, neighbour)

                if key in visited_edges:
                    continue

                path = [start, neighbour]
                visited_edges.add(key)

                previous = start
                current = neighbour

                while current not in critical_nodes:
                    next_nodes = [
                        node
                        for node in component_graph.neighbors(current)
                        if node != previous
                        and edge_key(current, node) not in visited_edges
                    ]

                    if not next_nodes:
                        break

                    next_node = next_nodes[0]

                    visited_edges.add(
                        edge_key(current, next_node)
                    )

                    path.append(next_node)
                    previous, current = current, next_node

                add_polyline(path)

        # Handle closed loops with no endpoints or junctions.
        # These components consist entirely of degree-2 nodes.
        for source, target in component_graph.edges():
            key = edge_key(source, target)

            if key in visited_edges:
                continue

            path = [source, target]
            visited_edges.add(key)

            previous = source
            current = target

            while current != source:
                next_nodes = [
                    node
                    for node in component_graph.neighbors(current)
                    if node != previous
                    and edge_key(current, node) not in visited_edges
                ]

                if not next_nodes:
                    break

                next_node = next_nodes[0]

                visited_edges.add(
                    edge_key(current, next_node)
                )

                path.append(next_node)
                previous, current = current, next_node

            add_polyline(path)

    expected_edges = {
        frozenset((source, target))
        for source, target in graph.edges()
    }

    if visited_edges != expected_edges:
        missing_edges = expected_edges - visited_edges
        raise RuntimeError(
            f"VTP export omitted {len(missing_edges)} graph edges."
        )
        
    # Export isolated nodes as vertices.
    for node in graph.nodes:
        if graph.degree(node) == 0:
            vertex = vtk.vtkVertex()
            vertex.GetPointIds().SetId(
                0, node_to_point[node]
            )
            vertices.InsertNextCell(vertex)

    # Create VTK PolyData.
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.SetVerts(vertices)
    
    # Determine the physical bounds of the graph.
    bounds = polydata.GetBounds()

    padding = 2.0 * radius_nm

    model_bounds = (
        bounds[0] - padding,
        bounds[1] + padding,
        bounds[2] - padding,
        bounds[3] + padding,
        bounds[4] - padding,
        bounds[5] + padding,
    )

    # Choose approximately isotropic sampling in physical space.
    spacing = radius_nm / resolution

    dimensions = [
        max(
            2,
            int(np.ceil(
                (model_bounds[2 * axis + 1]
                 - model_bounds[2 * axis]) / spacing
            )) + 1,
        )
        for axis in range(3)
    ]

    max_dimension = 500

    if max(dimensions) > max_dimension:
        raise ValueError(
            f"Implicit modeller dimensions {dimensions} are too large. "
            "Increase radius_nm or reduce resolution."
        )
        
    # Build the distance field around the dislocation network.
    modeller = vtk.vtkImplicitModeller()
    modeller.SetInputData(polydata)
    modeller.SetModelBounds(*model_bounds)
    modeller.SetSampleDimensions(*dimensions)
    model_diagonal = np.sqrt(
        (model_bounds[1] - model_bounds[0]) ** 2
        + (model_bounds[3] - model_bounds[2]) ** 2
        + (model_bounds[5] - model_bounds[4]) ** 2
    )
    
    max_distance_nm = 2.0 * radius_nm
    max_distance_percent = (
        100.0 * max_distance_nm / model_diagonal
    )
    
    modeller.SetMaximumDistance(max_distance_percent)
    modeller.Update()

    # Extract the constant-radius surface.
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(modeller.GetOutputPort())
    contour.SetValue(0, radius_nm)
    contour.Update()

    # Improve surface normals and visual smoothness.
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(contour.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    # Save the resulting tubular surface.
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(filename))
    writer.SetInputConnection(normals.GetOutputPort())
    
    writer.SetDataModeToBinary()
    
    if hasattr(writer, "SetEncodeAppendedData"):
        writer.SetEncodeAppendedData(False)
    
    if writer.Write() != 1:
        raise IOError(
            f"Failed to write dislocation surface to {filename}"
        )