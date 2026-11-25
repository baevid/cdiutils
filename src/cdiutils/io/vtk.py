import numpy as np

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