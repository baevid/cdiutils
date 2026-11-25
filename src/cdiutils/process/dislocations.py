"""Dislocation detection and graph analysis for 3D BCDI phase reconstructions.

The detector evaluates the wrapped phase circulation around elementary
plaquettes in the three orthogonal planes. Plaquettes with a circulation
larger than a configurable threshold are mapped onto a 3D candidate mask,
restricted to the crystal support, and skeletonised. A NetworkX graph is
then built from the skeleton using 26-neighbour connectivity.

The graph intentionally retains every skeleton voxel as a node. This keeps
the result transparent and makes it possible to add node reduction,
endpoint connection, and crystallographic categorisation as later steps.

This procedure was adapted from https://doi.org/10.1002/adma.202500221
"""

from __future__ import annotations

from itertools import product
from typing import Any

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def _wrapped_difference(delta: np.ndarray) -> np.ndarray:
    """Return phase differences wrapped to [-pi, pi]."""
    return np.angle(np.exp(1j * delta))


def _ring_slices(
    ndim: int,
    axis_a: int,
    axis_b: int,
) -> tuple[tuple[slice, ...], tuple[tuple[slice, ...], ...]]:
    """Return the centre slice and ordered perimeter slices of a 3x3 loop."""
    centre = [slice(None)] * ndim
    centre[axis_a] = slice(1, -1)
    centre[axis_b] = slice(1, -1)
    centre = tuple(centre)

    # Clockwise perimeter around the central voxel in the selected plane.
    offsets = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
    )
    perimeter = []
    for offset_a, offset_b in offsets:
        current = [slice(None)] * ndim
        for axis, offset in ((axis_a, offset_a), (axis_b, offset_b)):
            if offset == -1:
                current[axis] = slice(None, -2)
            elif offset == 0:
                current[axis] = slice(1, -1)
            else:
                current[axis] = slice(2, None)
        perimeter.append(tuple(current))
    return centre, tuple(perimeter)


def phase_circulation(
    phase: np.ndarray,
    support: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate phase circulation around 3x3 loops in the three planes.

    The returned array has shape ``(3, *phase.shape)``. Each circulation
    value is assigned to the central voxel of its 3x3 loop. Loops are
    retained only when the centre and all eight perimeter voxels are
    inside ``support``. Array axes are interpreted in native NumPy order;
    the three plane pairs are (0, 1), (0, 2), and (1, 2).
    """
    phase = np.asarray(phase, dtype=float)
    if phase.ndim != 3:
        raise ValueError(f"phase must be 3D, got {phase.ndim}D.")
    if not np.all(np.isfinite(phase)):
        raise ValueError(
            "phase must contain finite values. Unwrap or fill the phase "
            "before calculating dislocation circulation."
        )

    if support is None:
        support_array = np.ones(phase.shape, dtype=bool)
    else:
        support_array = np.asarray(support, dtype=bool)
        if support_array.shape != phase.shape:
            raise ValueError(
                "support and phase must have identical shapes, "
                f"got {support_array.shape} and {phase.shape}."
            )

    circulation = np.zeros((3,) + phase.shape, dtype=float)
    plane_axes = ((0, 1), (0, 2), (1, 2))

    for plane_index, (axis_a, axis_b) in enumerate(plane_axes):
        centre_slice, perimeter_slices = _ring_slices(
            phase.ndim, axis_a, axis_b
        )
        perimeter_phase = [phase[slices] for slices in perimeter_slices]
        loop = np.zeros_like(perimeter_phase[0], dtype=float)
        valid = support_array[centre_slice].copy()

        for index, current_phase in enumerate(perimeter_phase):
            next_phase = perimeter_phase[(index + 1) % len(perimeter_phase)]
            loop += _wrapped_difference(next_phase - current_phase)
            valid &= support_array[perimeter_slices[index]]

        circulation[(plane_index,) + centre_slice] = np.where(
            valid, loop, 0.0
        )

    return circulation


def build_dislocation_graph(
    skeleton: np.ndarray,
    voxel_size: tuple[float, float, float] | list[float] | np.ndarray,
    min_length_nm: float = 0.0,
    skeletonization_fallback: np.ndarray | None = None,
) -> tuple[nx.Graph, np.ndarray]:
    """Build a 26-neighbour graph from a 3D skeleton.

    Nodes represent individual skeleton voxels. ``array_index`` is in
    NumPy axis order, while ``xyz_nm`` is in reversed (x, y, z) order.
    Edge weights are physical distances in nm.
    """
    skeleton = np.asarray(skeleton, dtype=bool)
    if skeleton.ndim != 3:
        raise ValueError("skeleton must be a 3D array.")

    voxel_size = np.asarray(voxel_size, dtype=float)
    if voxel_size.shape != (3,) or np.any(~np.isfinite(voxel_size)):
        raise ValueError("voxel_size must contain three finite values.")
    if np.any(voxel_size <= 0):
        raise ValueError("voxel_size values must be positive.")
    if min_length_nm < 0:
        raise ValueError("min_length_nm must be non-negative.")
    if skeletonization_fallback is None:
        fallback_mask = np.zeros_like(skeleton, dtype=bool)
    else:
        fallback_mask = np.asarray(skeletonization_fallback, dtype=bool)
        if fallback_mask.shape != skeleton.shape:
            raise ValueError(
                "skeletonization_fallback must match the skeleton shape."
            )

    coordinates = np.argwhere(skeleton)
    graph = nx.Graph()
    graph.graph.update(
        voxel_size_nm=voxel_size.tolist(),
        connectivity=26,
        coordinate_convention="xyz_nm (x, y, z); array_index (axis 0, 1, 2)",
    )

    coordinate_to_id = {
        tuple(coordinate): point_id
        for point_id, coordinate in enumerate(coordinates)
    }

    for point_id, coordinate in enumerate(coordinates):
        array_index = tuple(int(value) for value in coordinate)
        xyz_nm = tuple(
            float(value)
            for value in (coordinate[::-1] * voxel_size[::-1])
        )
        graph.add_node(
            point_id,
            point_id=int(point_id),
            array_index=array_index,
            xyz_nm=xyz_nm,
        )

    neighbour_offsets = [
        offset
        for offset in product((-1, 0, 1), repeat=3)
        if offset != (0, 0, 0)
    ]
    for point_id, coordinate in enumerate(coordinates):
        for offset in neighbour_offsets:
            neighbour = tuple(
                int(value + delta)
                for value, delta in zip(coordinate, offset)
            )
            neighbour_id = coordinate_to_id.get(neighbour)
            if neighbour_id is None or neighbour_id <= point_id:
                continue
            delta_nm = np.asarray(offset, dtype=float) * voxel_size
            graph.add_edge(
                point_id,
                neighbour_id,
                length_nm=float(np.linalg.norm(delta_nm)),
            )

    filtered_skeleton = np.zeros_like(skeleton)
    dislocation_id = 0
    for component in nx.connected_components(graph):
        component_graph = graph.subgraph(component)
        component_used_fallback = any(
            fallback_mask[tuple(graph.nodes[node_id]["array_index"])]
            for node_id in component
        )

        if component_used_fallback:
            # A component preserved after failed thinning is a candidate
            # cluster, not a true skeleton. Use its principal-axis extent
            # only as a rough size estimate; do not interpret it as line
            # length.
            coordinates_nm = np.asarray(
                [
                    graph.nodes[node_id]["xyz_nm"]
                    for node_id in component
                ],
                dtype=float,
            )
            centred = coordinates_nm - coordinates_nm.mean(axis=0)
            if len(component) > 1:
                _, _, principal_axes = np.linalg.svd(
                    centred, full_matrices=False
                )
                projection = centred @ principal_axes[0]
                component_length = float(
                    projection.max() - projection.min()
                )
            else:
                component_length = 0.0
            length_method = "principal_axis_extent_approximation"
        else:
            # Use a minimum spanning tree to avoid counting redundant
            # diagonal links in the 26-neighbour voxel graph as line length.
            spanning_tree = nx.minimum_spanning_tree(
                component_graph, weight="length_nm"
            )
            component_length = float(
                sum(
                    data["length_nm"]
                    for _, _, data in spanning_tree.edges(data=True)
                )
            )
            length_method = "skeleton_graph_minimum_spanning_tree"

        if component_length < min_length_nm:
            graph.remove_nodes_from(component)
            continue

        degrees = [degree for _, degree in component_graph.degree()]
        if component_used_fallback:
            dislocation_type = "unreduced_candidate_cluster"
        elif not degrees or max(degrees) == 0:
            dislocation_type = "isolated"
        elif all(degree == 2 for degree in degrees):
            dislocation_type = "loop"
        elif max(degrees) > 2:
            dislocation_type = "branched"
        else:
            dislocation_type = "segment"

        for node_id in component:
            node = graph.nodes[node_id]
            array_index = tuple(node["array_index"])
            filtered_skeleton[array_index] = True
            node["num_neighbours"] = int(graph.degree[node_id])
            node["dislocation_id"] = int(dislocation_id)
            node["dislocation_type"] = dislocation_type
            node["component_length_nm"] = component_length
            node["component_length_method"] = length_method
            node["skeletonization_fallback"] = bool(
                fallback_mask[array_index]
            )

        dislocation_id += 1

    graph.graph["number_of_dislocations"] = dislocation_id
    graph.graph["number_of_skeleton_voxels"] = graph.number_of_nodes()
    graph.graph["skeletonization_fallback_voxels"] = int(
        np.count_nonzero(fallback_mask & filtered_skeleton)
    )
    return graph, filtered_skeleton


def identify_dislocations(
    phase: np.ndarray,
    support: np.ndarray,
    voxel_size: tuple[float, float, float] | list[float] | np.ndarray,
    circulation_threshold: float = 0.75 * np.pi,
    min_length_nm: float = 0.0,
) -> dict[str, Any]:
    """Detect dislocation candidates and construct a voxel graph.

    Args:
        phase: Unwrapped 3D phase in radians. Circulation is calculated
            from wrapped phase differences, so adding a smooth phase ramp
            does not change the topological winding.
        support: Boolean or binary crystal support. Plaquettes crossing
            the support boundary are excluded.
        voxel_size: Voxel spacing in nm, in NumPy array-axis order.
        circulation_threshold: Minimum absolute plaquette circulation
            in radians. The default is 0.75*pi.
        min_length_nm: Remove graph components shorter than this total
            graph-edge length. Defaults to 0 (no length filtering).

    Returns:
        Dictionary containing circulation, candidate mask, skeleton,
        NetworkX graph, and a compact summary.
    """
    if not np.isfinite(circulation_threshold) or circulation_threshold <= 0:
        raise ValueError("circulation_threshold must be positive and finite.")

    circulation = phase_circulation(phase, support)
    candidate_mask = (
        np.max(np.abs(circulation), axis=0) >= circulation_threshold
    )
    candidate_mask &= np.asarray(support, dtype=bool)

    # Lee's 3D skeletonisation method is used by scikit-image for 3D input.
    # Very thin components (e.g. a 2x2-voxel line) can be removed entirely
    # by Lee thinning. Process components independently and preserve such
    # components unchanged rather than silently losing a detected defect.
    labels, number_of_components = ndi.label(
        candidate_mask,
        structure=np.ones((3, 3, 3), dtype=bool),
    )
    skeleton = np.zeros_like(candidate_mask, dtype=bool)
    fallback_mask = np.zeros_like(candidate_mask, dtype=bool)
    fallback_components = 0
    for component_id in range(1, number_of_components + 1):
        component = labels == component_id
        component_skeleton = skeletonize(component, method="lee")
        if not np.any(component_skeleton):
            component_skeleton = component
            fallback_mask |= component
            fallback_components += 1
        skeleton |= component_skeleton

    graph, skeleton = build_dislocation_graph(
        skeleton,
        voxel_size=voxel_size,
        min_length_nm=min_length_nm,
        skeletonization_fallback=fallback_mask,
    )

    summary = {
        "candidate_voxels": int(np.count_nonzero(candidate_mask)),
        "skeleton_voxels": int(np.count_nonzero(skeleton)),
        "number_of_dislocations": int(graph.graph["number_of_dislocations"]),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "skeletonization_fallback_components": int(fallback_components),
    }
    return {
        "dislocation_circulation": circulation,
        "dislocation_mask": candidate_mask.astype(np.uint8),
        "dislocation_skeleton": skeleton.astype(np.uint8),
        "dislocation_graph": graph,
        "dislocation_summary": summary,
    }
