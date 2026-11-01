"""Map geometry for the corridor: real streets from a cached OpenStreetMap build, or a schematic layout.

The CTM is one-dimensional; coordinates are for drawing only. With `street_geometry_path` set (see
scripts/build_street_geometry.py) every CTM link is laid along the real road segment between two real
intersections and each cross street along a real side road. Without it, the arterial runs west to east through
the configured centre and cross streets meet each junction alternately from north and south.
"""

import json
import math
from pathlib import Path
from typing import Any

from citysense.config.schemas import NetworkConfig
from citysense.config.units import EARTH_RADIUS_M, HALF_FACTOR, METERS_PER_DEGREE_LAT
from citysense.live.config import LiveConfig
from citysense.network.corridor import CorridorNetwork, cross_leg_cells


# ---- polyline helpers -------------------------------------------------------------------------------
def haversine_m(a: list[float], b: list[float]) -> float:
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def cumulative_m(poly: list[list[float]]) -> list[float]:
    out = [0.0]
    for i in range(1, len(poly)):
        out.append(out[-1] + haversine_m(poly[i - 1], poly[i]))
    return out


def point_at(poly: list[list[float]], cum: list[float], d: float) -> list[float]:
    d = min(max(d, 0.0), cum[-1])
    for i in range(1, len(poly)):
        if cum[i] >= d:
            seg = cum[i] - cum[i - 1]
            t = 0.0 if seg <= 0.0 else (d - cum[i - 1]) / seg
            return [poly[i - 1][0] + (poly[i][0] - poly[i - 1][0]) * t, poly[i - 1][1] + (poly[i][1] - poly[i - 1][1]) * t]
    return list(poly[-1])


def slice_polyline(poly: list[list[float]], cum: list[float], d0: float, d1: float) -> list[list[float]]:
    pts = [point_at(poly, cum, d0)]
    pts += [list(p) for p, c in zip(poly, cum) if d0 < c < d1]
    pts.append(point_at(poly, cum, d1))
    return pts


def split_evenly(poly: list[list[float]], parts: int, start: float = 0.0, end: float | None = None) -> list[list[list[float]]]:
    cum = cumulative_m(poly)
    stop = cum[-1] if end is None else end
    step = (stop - start) / parts
    return [slice_polyline(poly, cum, start + k * step, start + (k + 1) * step) for k in range(parts)]


# ---- layouts ----------------------------------------------------------------------------------------
def build_geometry(net: CorridorNetwork, net_cfg: NetworkConfig, live_cfg: LiveConfig) -> dict[str, Any]:
    cells_per_link = net_cfg.num_cells_per_link
    street_file = Path(live_cfg.street_geometry_path) if live_cfg.street_geometry_path else None
    if street_file is not None and street_file.is_file():
        with open(street_file, "r", encoding="utf-8") as f:
            streets = json.load(f)
        if len(streets["junction_distances_m"]) == len(net.junction_cells):
            cells, junctions, meta = _street_layout(net, cells_per_link, streets)
            return _finish(net, cells_per_link, cells, junctions, schematic=False, streets=meta)
    cells, junctions = _schematic_layout(net, net_cfg, live_cfg, cells_per_link)
    return _finish(net, cells_per_link, cells, junctions, schematic=True, streets=None)


def _street_layout(
    net: CorridorNetwork, cells_per_link: int, streets: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    art = streets["arterial"]
    marks = [0.0] + list(streets["junction_distances_m"]) + [cumulative_m(art)[-1]]
    cells: list[dict[str, Any]] = []
    for link in range(len(marks) - 1):
        pieces = split_evenly(art, cells_per_link, marks[link], marks[link + 1])
        for k, coords in enumerate(pieces):
            c = link * cells_per_link + k
            cells.append({"id": int(c), "kind": "arterial", "link": int(link), "coords": coords})
    junctions: list[dict[str, Any]] = []
    for j, jc in sorted(net.junction_cells.items()):
        legs = streets["cross_streets"][j]
        legs = legs if isinstance(legs, list) else [legs]  # older single-leg files
        for leg in range(len(jc["cross_approaches"])):
            street = cross_leg_cells(net, j, leg)
            leg_geo = legs[min(leg, len(legs) - 1)]
            pieces = split_evenly(leg_geo["path"], len(street))
            synthetic = str(leg_geo.get("method", "")).startswith("straight")
            for c, coords in zip(street, pieces):
                cells.append({"id": int(c), "kind": "cross", "link": int(j), "leg": leg, "synthetic": synthetic, "coords": coords})
        loc = streets["junction_locations"][j]
        junctions.append(_junction(net, j, jc, loc[0], loc[1]))
    meta = {
        "arterial_name": streets.get("arterial_name"),
        "attribution": streets.get("attribution"),
        "source": streets.get("source"),
        "link_scale": streets.get("link_scale"),
    }
    return cells, junctions, meta


def _schematic_layout(
    net: CorridorNetwork, net_cfg: NetworkConfig, live_cfg: LiveConfig, cells_per_link: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lat0 = live_cfg.map_center_lat if live_cfg.map_center_lat is not None else net_cfg.center_lat
    lon0 = live_cfg.map_center_lon if live_cfg.map_center_lon is not None else net_cfg.center_lon
    m_per_deg_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(lat0))

    def to_ll(x_m: float, y_m: float) -> list[float]:
        return [lat0 + y_m / METERS_PER_DEGREE_LAT, lon0 + x_m / m_per_deg_lon]

    arterial = net.arterial_cells
    total_len = float(sum(net.cell_length[c] for c in arterial))
    x = -total_len * HALF_FACTOR
    cell_x_end: dict[int, float] = {}
    cells: list[dict[str, Any]] = []
    for c in arterial:
        x_next = x + float(net.cell_length[c])
        cells.append({"id": int(c), "kind": "arterial", "link": int(c // cells_per_link), "coords": [to_ll(x, 0.0), to_ll(x_next, 0.0)]})
        cell_x_end[int(c)] = x_next
        x = x_next
    junctions: list[dict[str, Any]] = []
    for j, jc in sorted(net.junction_cells.items()):
        art_c = jc["arterial_approach"]
        jx = cell_x_end[art_c]
        n_legs = len(jc["cross_approaches"])
        for leg in range(n_legs):
            side = (1.0 if j % 2 == 0 else -1.0) if n_legs == 1 else (1.0 if leg == 0 else -1.0)
            street = cross_leg_cells(net, j, leg)
            y = float(sum(net.cell_length[c] for c in street))
            for c in street:
                y_next = y - float(net.cell_length[c])
                cells.append({"id": int(c), "kind": "cross", "link": int(j), "leg": leg,
                              "coords": [to_ll(jx, side * y), to_ll(jx, side * y_next)]})
                y = y_next
        lat, lon = to_ll(jx, 0.0)
        junctions.append(_junction(net, j, jc, lat, lon))
    return cells, junctions


def _junction(net: CorridorNetwork, j: int, jc: dict[str, int], lat: float, lon: float) -> dict[str, Any]:
    art_c = jc["arterial_approach"]
    cross_c = jc["cross_approach"]
    return {
        "id": int(j),
        "lat": lat,
        "lon": lon,
        "arterial_approach": int(art_c),
        "cross_approach": int(cross_c),
        "cross_approaches": [int(c) for c in jc["cross_approaches"]],
        "arterial_phase": int(net.signal_controlled_cells[art_c][1]),
        "cross_phase": int(net.signal_controlled_cells[cross_c][1]),
    }


def _finish(
    net: CorridorNetwork,
    cells_per_link: int,
    cells: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    schematic: bool,
    streets: dict[str, Any] | None,
) -> dict[str, Any]:
    sources = source_paths(net, cells_per_link)
    cell_coords = {cell["id"]: cell["coords"] for cell in cells}
    source_list = [
        {
            "id": sid,
            "kind": src["kind"],
            "junction": src["junction"],
            "leg": src.get("leg"),
            "cell": src["path"][0],
            "name": src["name"],
            "lat": cell_coords[src["path"][0]][0][0],
            "lon": cell_coords[src["path"][0]][0][1],
        }
        for sid, src in sources.items()
    ]
    all_pts = [p for cell in cells for p in cell["coords"]]
    lats = [p[0] for p in all_pts]
    lons = [p[1] for p in all_pts]
    return {
        "center": [sum(lats) / len(lats), sum(lons) / len(lons)],
        "bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],
        "cells": sorted(cells, key=lambda cell: cell["id"]),
        "junctions": junctions,
        "sources": source_list,
        "cell_length": [float(v) for v in net.cell_length],
        "schematic": schematic,
        "streets": streets,
    }


def source_paths(net: CorridorNetwork, cells_per_link: int) -> dict[int, dict[str, Any]]:
    """Entry points where trips can start, each with the ordered cell path a platoon follows.

    Source 0 is the arterial's entry gateway; then one source per side-road leg of every junction.
    """
    sources: dict[int, dict[str, Any]] = {
        0: {"kind": "arterial", "junction": None, "leg": None, "name": "Arterial gateway", "path": list(net.arterial_cells)},
    }
    sid = 1
    for j, jc in sorted(net.junction_cells.items()):
        n_legs = len(jc["cross_approaches"])
        for leg in range(n_legs):
            side = "" if n_legs == 1 else (" north" if leg == 0 else " south")
            sources[sid] = {"kind": "cross", "junction": int(j), "leg": leg,
                            "name": f"Cross street J{j + 1}{side}", "path": cross_leg_cells(net, j, leg)}
            sid += 1
    return sources
