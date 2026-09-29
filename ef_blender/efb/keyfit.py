"""Which of a densely sampled curve's keys have to stay for it to still describe the pose.

A baked IK handle carries one key per whole frame, which is correct and unusable: an
animator cannot work in a channel with a key on every frame. Blender lerps between
keyframes and so does Epic Fight, so the samples between two kept keys are already
predicted by the pair - keeping one only buys whatever the prediction gets wrong.

`fit_indices` is Douglas-Peucker over a polyline whose components are pre-divided by
their own tolerance, so one max-norm bound covers channels measured in metres, quaternion
units and scale at once. Endpoints are always kept, and every dropped sample is inside
tolerance of the line the survivors draw - that is the guarantee, not an average.
"""

from __future__ import annotations

__all__ = ["fit_indices", "normalised", "hemisphere_rows", "constant"]


def constant(values, eps=1e-9) -> bool:
    return not values or (max(values) - min(values)) <= eps


def normalised(rows, tolerances) -> list:
    """Each row's components divided by their tolerance, so 1.0 is the bound everywhere."""
    inv = [1.0 / t if t else 0.0 for t in tolerances]
    return [[v * s for v, s in zip(row, inv)] for row in rows]


def hemisphere_rows(rows, spans) -> list:
    """Negate whole quaternions that turned over between samples.

    A quaternion and its negation are the same rotation, but Blender lerps the four
    components one by one, so a sign flip between two kept keys would swing the pose out
    through zero. `spans` is [(start, length)] naming the quaternion blocks in each row.
    """
    out = [list(rows[0])] if rows else []
    for row in rows[1:]:
        row = list(row)
        for start, length in spans:
            prev = out[-1][start:start + length]
            here = row[start:start + length]
            if sum(a * b for a, b in zip(prev, here)) < 0.0:
                row[start:start + length] = [-v for v in here]
        out.append(row)
    return out


def _deviation(rows, times, lo, hi):
    """Worst max-norm distance from the chord lo..hi, and where. Ends are exact."""
    t0, t1 = times[lo], times[hi]
    span = t1 - t0
    worst, at = 0.0, lo
    for i in range(lo + 1, hi):
        u = 0.0 if span == 0.0 else (times[i] - t0) / span
        row = rows[i]
        d = 0.0
        for c, v in enumerate(row):
            got = rows[lo][c] + (rows[hi][c] - rows[lo][c]) * u
            d = max(d, abs(v - got))
        if d > worst:
            worst, at = d, i
    return worst, at


def fit_indices(times, rows, tol=1.0) -> list:
    """Sorted indices to keep. `rows` are already normalised; `tol` is the max-norm bound.

    Iterative rather than recursive: a 700-frame clip would otherwise sit 700 deep on a
    curve that only steps.
    """
    n = len(rows)
    if n <= 2:
        return list(range(n))
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        worst, at = _deviation(rows, times, lo, hi)
        if worst > tol:
            keep.add(at)
            stack.append((lo, at))
            stack.append((at, hi))
    return sorted(keep)
