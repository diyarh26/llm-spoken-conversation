"""
Route geometry and the task-success score (torch-free).

Coordinates follow data/maptask/maps.json: x grows east (right), y grows SOUTH (down),
as on the paper maps and in image pixels.

Task success mirrors the corpus's own path deviation score (devscore: the area between
the giver's route and the follower's drawn route). We cannot measure ink on paper, so a
separate, fresh model call reads the finished dialogue with ONLY the follower's map and
draws the route the dialogue describes (see maptask.prompts.build_reconstruction). The
same reader is applied to the 128 human dialogues, so humans and models are scored by
the identical procedure. Score = mean distance (map units) between the two routes after
resampling both to the same number of evenly spaced points; 0 is a perfect copy.
"""

from __future__ import annotations

import json
import math
import re

Point = tuple[float, float]


def _seg_project(p: Point, a: Point, b: Point) -> tuple[float, Point]:
    """Fraction along segment a->b of the closest point to p, and that point."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return t, (ax + t * dx, ay + t * dy)


def path_length(route: list[Point]) -> float:
    return sum(math.dist(route[i], route[i + 1]) for i in range(len(route) - 1))


def resample(route: list[Point], n: int = 100) -> list[Point]:
    """n points evenly spaced by arc length along the polyline (endpoints included)."""
    if len(route) == 1:
        return [tuple(route[0])] * n
    total = path_length(route)
    if total == 0:
        return [tuple(route[0])] * n
    out, seg, done = [], 0, 0.0
    for k in range(n):
        target = total * k / (n - 1)
        while seg < len(route) - 2 and done + math.dist(route[seg], route[seg + 1]) < target:
            done += math.dist(route[seg], route[seg + 1])
            seg += 1
        a, b = route[seg], route[seg + 1]
        length = math.dist(a, b)
        t = 0.0 if length == 0 else min(1.0, (target - done) / length)
        out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def route_deviation(drawn: list[Point], gold: list[Point], n: int = 100) -> float:
    """Mean distance between corresponding evenly spaced points of the two routes."""
    if not drawn:
        return float("nan")
    a, b = resample(drawn, n), resample(gold, n)
    return sum(math.dist(p, q) for p, q in zip(a, b, strict=True)) / n


def passes(route: list[Point], landmarks: list[dict], max_dist: float = 7.0) -> list[dict]:
    """Landmarks the route passes close to, in route order, with the side they are on as
    seen by someone travelling along the route: [{name, label, side, distance}].
    START and FINISH are left out. y grows downward, so a positive cross product means
    the landmark is on the traveller's RIGHT."""
    found = []
    for lm in landmarks:
        if lm["name"] in ("start", "finish"):
            continue
        p = (lm["x"], lm["y"])
        best = None
        for i in range(len(route) - 1):
            t, q = _seg_project(p, route[i], route[i + 1])
            d = math.dist(p, q)
            if best is None or d < best[0]:
                best = (d, i + t, route[i], route[i + 1], q)
        if best is None or best[0] > max_dist:
            continue
        d, pos, a, b, q = best
        cross = (b[0] - a[0]) * (p[1] - q[1]) - (b[1] - a[1]) * (p[0] - q[0])
        found.append(
            {
                "name": lm["name"],
                "label": lm.get("label", lm["name"]),
                "side": "right" if cross > 0 else "left",
                "distance": round(d, 1),
                "position": pos,
            }
        )
    found.sort(key=lambda f: f["position"])
    for f in found:
        del f["position"]
    return found


_PAIR = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse_points(text: str) -> list[Point]:
    """The [x, y] points in a reader's answer. Accepts a JSON list or any text that
    contains [x, y] pairs; returns [] when there are none."""
    try:
        data = json.loads(text.strip().strip("`").removeprefix("json").strip())
        if isinstance(data, list) and all(
            isinstance(p, list | tuple) and len(p) == 2 for p in data
        ):
            return [(float(x), float(y)) for x, y in data]
    except (ValueError, TypeError):
        pass
    return [(float(x), float(y)) for x, y in _PAIR.findall(text)]
