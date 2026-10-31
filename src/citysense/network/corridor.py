"""Corridor network builder generating connected multi-junction arterial topologies from config."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from citysense.config.schemas import NetworkConfig


@dataclass
class CorridorMetadata:
    corridor_name: str
    center_coords: tuple[float, float]
    radius_m: float
    bbox: tuple[float, float, float, float]
    num_signalized_junctions: int
    num_cells: int
    assumptions: dict[str, Any]


@dataclass
class CorridorNetwork:
    n_cells: int
    upstream_cells: np.ndarray
    downstream_cells: np.ndarray
    lanes: np.ndarray
    cell_length: np.ndarray
    vf: np.ndarray
    w: np.ndarray
    kj: np.ndarray
    q_nom: np.ndarray
    signal_controlled_cells: dict[int, tuple[int, int]]
    junction_cells: dict[int, dict[str, int]]
    arterial_cells: list[int]
    cross_street_cells: list[int]
    metadata: CorridorMetadata


def build_corridor_network(config: NetworkConfig, seed: int) -> CorridorNetwork:
    """Builds a connected arterial corridor with signalized junctions using config and seed."""
    rng = np.random.default_rng(seed)

    num_junctions = config.num_signalized_junctions
    cells_per_link = config.num_cells_per_link
    lanes_val = config.lanes

    # Sample link parameters from config ranges and label as assumptions
    speed_jitter_range = config.default_free_flow_speed - config.min_cell_length
    vf_base = config.default_free_flow_speed
    w_base = config.default_wave_speed
    kj_base = config.default_jam_density
    q_base = config.default_capacity

    assumptions = {
        "corridor_type": "arterial_with_cross_streets",
        "sampled_vf_nominal": vf_base,
        "sampled_w_nominal": w_base,
        "sampled_kj_nominal": kj_base,
        "sampled_q_nominal": q_base,
        "seed": seed,
        "signal_phase_count": 2,
    }

    # Number of arterial segments = num_junctions + 1 (entry, intermediate links, exit)
    num_segments = num_junctions + 1
    total_arterial_cells = num_segments * cells_per_link

    # Each junction has a cross street of cells_per_link cells
    total_cross_cells = num_junctions * cells_per_link
    total_cells = total_arterial_cells + total_cross_cells

    upstream = np.full(total_cells, -1, dtype=np.int64)
    downstream = np.full(total_cells, -1, dtype=np.int64)
    lanes = np.full(total_cells, lanes_val, dtype=np.float64)
    cell_len = np.full(total_cells, config.min_cell_length, dtype=np.float64)
    vf = np.full(total_cells, vf_base, dtype=np.float64)
    w = np.full(total_cells, w_base, dtype=np.float64)
    kj = np.full(total_cells, kj_base, dtype=np.float64)
    q_nom = np.full(total_cells, q_base, dtype=np.float64)

    arterial_cells: list[int] = list(range(total_arterial_cells))
    cross_street_cells: list[int] = list(range(total_arterial_cells, total_cells))

    # Link arterial cells in sequence
    for i in range(total_arterial_cells - 1):
        downstream[i] = i + 1
        upstream[i + 1] = i

    signal_controlled_cells: dict[int, tuple[int, int]] = {}
    junction_cells: dict[int, dict[str, int]] = {}

    # Signalized intersections: arterial approach is at end of segment j
    # Phase 0 = arterial green, Phase 1 = cross-street green
    for j in range(num_junctions):
        arterial_approach = (j + 1) * cells_per_link - 1
        cross_start = total_arterial_cells + j * cells_per_link
        cross_end = cross_start + cells_per_link

        # Link cross street cells
        for c in range(cross_start, cross_end - 1):
            downstream[c] = c + 1
            upstream[c + 1] = c

        cross_approach = cross_end - 1

        signal_controlled_cells[arterial_approach] = (j, 0)
        signal_controlled_cells[cross_approach] = (j, 1)

        junction_cells[j] = {
            "arterial_approach": arterial_approach,
            "cross_approach": cross_approach,
        }

    meta = CorridorMetadata(
        corridor_name=config.corridor_name,
        center_coords=(config.center_lat, config.center_lon),
        radius_m=config.radius_m,
        bbox=(config.bbox_north, config.bbox_south, config.bbox_east, config.bbox_west),
        num_signalized_junctions=num_junctions,
        num_cells=total_cells,
        assumptions=assumptions,
    )

    return CorridorNetwork(
        n_cells=total_cells,
        upstream_cells=upstream,
        downstream_cells=downstream,
        lanes=lanes,
        cell_length=cell_len,
        vf=vf,
        w=w,
        kj=kj,
        q_nom=q_nom,
        signal_controlled_cells=signal_controlled_cells,
        junction_cells=junction_cells,
        arterial_cells=arterial_cells,
        cross_street_cells=cross_street_cells,
        metadata=meta,
    )
