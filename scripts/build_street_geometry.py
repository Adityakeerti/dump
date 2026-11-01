"""Build real-street geometry for the live twin's map from OpenStreetMap (via the public OSRM router).

Run once (needs internet); the result is cached in config/street_geometry.json and the twin never calls the
network itself:

    .venv/bin/python scripts/build_street_geometry.py                          # Rajpur Road, Dehradun (default)
    .venv/bin/python scripts/build_street_geometry.py --from 30.3245,78.0419 --to 30.3340,78.0530

What it does
1. Routes the arterial from --from to --to (driving profile) and keeps the full polyline.
2. Takes OSRM's intersections along the route (nodes where >= 3 roads meet) and picks one real intersection
   per signalised junction, as close as possible to evenly spaced along the route.
3. For each junction, picks the side road most perpendicular to the arterial and routes from a point
   --cross-m metres along it back to the junction (falls back to foot routing for one-way streets, then to a
   straight line), so cross streets follow real roads too.

The CTM physics are unchanged: each CTM link is drawn along the real segment between two chosen
intersections, so the drawing scale differs per link (reported as link_scale = drawn metres / CTM metres).
"""

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from citysense.config.loader import load_network_config  # noqa: E402

OSRM_DRIVE = "https://router.project-osrm.org/route/v1/driving"
OSRM_FOOT = "https://routing.openstreetmap.de/routed-foot/route/v1/foot"
R = 6371000.0


def hav(a, b):
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def offset(pt, bearing_deg, dist_m):
    la, lo = map(math.radians, pt)
    b = math.radians(bearing_deg)
    d = dist_m / R
    la2 = math.asin(math.sin(la) * math.cos(d) + math.cos(la) * math.sin(d) * math.cos(b))
    lo2 = lo + math.atan2(math.sin(b) * math.sin(d) * math.cos(la), math.cos(d) - math.sin(la) * math.sin(la2))
    return [math.degrees(la2), math.degrees(lo2)]


def route(base, a, b, steps=False):
    url = f"{base}/{a[1]},{a[0]};{b[1]},{b[0]}?overview=full&geometries=geojson&steps={'true' if steps else 'false'}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "citysense-twin/1.0"}), timeout=30) as r:
        d = json.load(r)
    if d.get("code") != "Ok":
        raise RuntimeError(f"OSRM: {d.get('code')} {d.get('message')}")
    return d["routes"][0]


def cumulative(poly):
    out = [0.0]
    for i in range(1, len(poly)):
        out.append(out[-1] + hav(poly[i - 1], poly[i]))
    return out


def point_at_dist(poly, cum, d):
    for i in range(1, len(poly)):
        if cum[i] >= d:
            t = (d - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
            return [poly[i - 1][0] + (poly[i][0] - poly[i - 1][0]) * t, poly[i - 1][1] + (poly[i][1] - poly[i - 1][1]) * t]
    return poly[-1]


def project(poly, cum, pt):
    """Arc-length position of the vertex nearest to pt."""
    i = min(range(len(poly)), key=lambda k: hav(poly[k], pt))
    return cum[i], i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="30.3245,78.0419", help="lat,lon of the arterial start (Clock Tower)")
    ap.add_argument("--to", dest="dst", default="30.3410,78.0590", help="lat,lon of the arterial end (Rajpur Road)")
    ap.add_argument("--cross-m", type=float, default=None, help="drawn length of each side-road leg (default: simulated leg length)")
    ap.add_argument("--stagger-m", type=float, default=60.0, help="use a side road this close along the arterial for a missing side")
    ap.add_argument("--out", default=str(ROOT / "config" / "street_geometry.json"))
    args = ap.parse_args()

    net = load_network_config(ROOT / "config" / "network.yaml")
    n_j = net.num_signalized_junctions
    n_legs = net.cross_street_legs
    ctm_link_m = net.num_cells_per_link * net.min_cell_length
    cross_m = args.cross_m or (net.cross_cells_per_leg or net.num_cells_per_link) * net.min_cell_length

    a = [float(x) for x in args.src.split(",")]
    b = [float(x) for x in args.dst.split(",")]
    r = route(OSRM_DRIVE, a, b, steps=True)
    poly = [[c[1], c[0]] for c in r["geometry"]["coordinates"]]
    cum = cumulative(poly)
    total = cum[-1]
    name = next((s["name"] for s in r["legs"][0]["steps"] if s.get("name")), "arterial")

    candidates = []
    for step in r["legs"][0]["steps"]:
        for it in step["intersections"]:
            if len(it.get("bearings", [])) < 3:
                continue
            loc = [it["location"][1], it["location"][0]]
            d, _ = project(poly, cum, loc)
            if 0.05 * total < d < 0.95 * total:
                candidates.append({"loc": loc, "dist": d, "bearings": it["bearings"], "in": it.get("in"), "out": it.get("out")})
    if len(candidates) < n_j:
        raise SystemExit(f"Only {len(candidates)} intersections on this route; need {n_j}. Pick a longer route.")

    def side_of(c, bearing):
        """'left' / 'right' of the arterial's direction of travel at intersection c."""
        out_b = c["bearings"][c["out"]] if c["out"] is not None else 0
        rel = (bearing - out_b) % 360
        return "right" if 20 < rel < 160 else "left" if 200 < rel < 340 else None

    def side_roads(c):
        return [(bb, side_of(c, bb)) for i, bb in enumerate(c["bearings"]) if i not in (c["in"], c["out"]) and side_of(c, bb)]

    # Prefer real four-way crossings, then T-junctions, close to even spacing.
    chosen, used = [], set()
    for k in range(1, n_j + 1):
        target = k * total / (n_j + 1)
        pool = [(i, c) for i, c in enumerate(candidates) if i not in used and (not chosen or c["dist"] > chosen[-1]["dist"] + 40)]
        i, best = min(pool, key=lambda ic: abs(ic[1]["dist"] - target) - (40 if len({s for _, s in side_roads(ic[1])}) == 2 else 0))
        used.add(i)
        chosen.append(best)

    def route_leg(far, end):
        for base, label in ((OSRM_DRIVE, "osrm-driving"), (OSRM_FOOT, "osrm-foot")):
            try:
                rr = route(base, far, end)
            except Exception as exc:
                print(f"    {label} failed ({exc})")
                continue
            if rr["distance"] <= 2.2 * cross_m + 2 * args.stagger_m:
                return [[p[1], p[0]] for p in rr["geometry"]["coordinates"]], label
        return None, None

    cross = []
    for jn, c in enumerate(chosen):
        out_b = c["bearings"][c["out"]] if c["out"] is not None else 0
        legs = []
        for side in (["left", "right"][:n_legs] if n_legs > 1 else [None]):
            own = [bb for bb, sd in side_roads(c) if side is None or sd == side]
            host, bearing, how = c, None, None
            if own:
                bearing = max(own, key=lambda bb: abs(math.sin(math.radians(bb - out_b))))
            else:  # staggered junction: nearest side road on that side within stagger_m along the arterial
                near = [(abs(o["dist"] - c["dist"]), o, bb) for o in candidates if abs(o["dist"] - c["dist"]) <= args.stagger_m
                        for bb, sd in side_roads(o) if sd == side]
                if near:
                    _, host, bearing = min(near, key=lambda x: x[0])
            if bearing is not None:
                far = offset(host["loc"], bearing, cross_m)
                path, how = route_leg(far, host["loc"])
                if path is not None and host is not c:  # continue along the arterial to this junction's stop line
                    d0, d1 = sorted((host["dist"], c["dist"]))
                    i0, i1 = project(poly, cum, point_at_dist(poly, cum, d0))[1], project(poly, cum, point_at_dist(poly, cum, d1))[1]
                    seg = poly[i0:i1 + 1] if host["dist"] < c["dist"] else list(reversed(poly[i0:i1 + 1]))
                    path = path + seg
                    how += "+stagger"
            if bearing is None or path is None:
                rb = (out_b + (-90 if side == "left" else 90)) % 360
                path, how = [offset(c["loc"], rb, cross_m), c["loc"]], "straight (no real road on this side)"
            path[-1] = c["loc"]
            legs.append({"path": [[round(p[0], 7), round(p[1], 7)] for p in path], "method": how, "side": side,
                         "length_m": round(cumulative(path)[-1], 1)})
            print(f"  J{jn + 1} {side or ''}: {how}, {legs[-1]['length_m']} m")
        cross.append(legs)

    dists = [0.0] + [c["dist"] for c in chosen] + [total]
    out = {
        "source": "OpenStreetMap data via OSRM (router.project-osrm.org, routing.openstreetmap.de)",
        "attribution": "© OpenStreetMap contributors",
        "arterial_name": name,
        "from": a,
        "to": b,
        "arterial": [[round(p[0], 7), round(p[1], 7)] for p in poly],
        "arterial_length_m": round(total, 1),
        "junction_distances_m": [round(c["dist"], 1) for c in chosen],
        "junction_locations": [c["loc"] for c in chosen],
        "cross_streets": cross,  # per junction: list of legs (leg 0 = left of travel direction, leg 1 = right)
        "link_scale": [round((dists[i + 1] - dists[i]) / ctm_link_m, 2) for i in range(n_j + 1)],
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{name}: {total:.0f} m, junctions at {out['junction_distances_m']} m, link scale {out['link_scale']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
