"""Control shapes as plain vertex/edge data. No bpy.

Bone space: +Y runs along the bone, so a ring around a bone lives in the XZ plane and a
shape lying on the ground belongs to a bone that points up (Root and its controls do).
Every widget is unit-sized; the builder scales it per bone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Widget", "WIDGETS", "widget", "circle_xz", "ring_loop"]


@dataclass(frozen=True)
class Widget:
    name: str
    verts: tuple
    edges: tuple


def ring_loop(n):
    return tuple((i, (i + 1) % n) for i in range(n))


def circle_xz(n=16, radius=1.0, y=0.0):
    verts = tuple((radius * math.cos(2 * math.pi * i / n), y,
                   radius * math.sin(2 * math.pi * i / n)) for i in range(n))
    return verts, ring_loop(n)


def _box(size=1.0):
    h = size * 0.5
    verts = tuple((x * h, y * h, z * h)
                  for x in (-1, 1) for y in (-1, 1) for z in (-1, 1))
    edges = ((0, 1), (1, 3), (3, 2), (2, 0),
             (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
    return verts, edges


def _diamond(size=1.0):
    h = size * 0.5
    verts = ((h, 0, 0), (0, h, 0), (-h, 0, 0), (0, -h, 0), (0, 0, h), (0, 0, -h))
    edges = ((0, 1), (1, 2), (2, 3), (3, 0),
             (0, 4), (1, 4), (2, 4), (3, 4),
             (0, 5), (1, 5), (2, 5), (3, 5))
    return verts, edges


def _square_xz(size=1.0):
    h = size * 0.5
    verts = ((-h, 0, -h), (h, 0, -h), (h, 0, h), (-h, 0, h))
    return verts, ((0, 1), (1, 2), (2, 3), (3, 0))


def _socket(size=1.0):
    """Weapon slot drawn one way along the mesh's own +Z, and turned onto the blade by
    rigdef.WIDGET_OFFSETS from rigedit.TOOL_AXIS - the tool axis is the socket bone's -Z.

    Not one vertex sits at negative z in the mesh's own frame, so the shape reaches
    0.329 m out of the grip and 0.000 m behind it. That beats being merely lopsided: a
    silhouette with a short end and a long end still has to be compared with itself before
    it says anything, and from the wrong angle the comparison is foreshortened away.

    The old "+Z is measured" here was measured in the HAND frame and then applied to the
    TOOL bone, and Tool_*.rest_local is a 179 degree flip about local X - which is exactly
    the difference. Remeasured over the same 47 item models: mean (-0.0110, -0.0841,
    +0.9395) in the Hand_R frame and (-0.0110, +0.0671, -0.9409) in Tool_R's, both with a
    resultant of 0.9433. The mesh keeps its +Z and the bone gets the half turn, so the
    shape stays shared and unmirrored.

    Three walls, all against other controls, and all still live:
      - nothing may run along the grip at y 0: that bar passed 0.035 from the IK hand
        cube against a 0.040 floor, which is why the rail starts 0.50 out and the pin,
        short enough to survive at 0.110, is what anchors the shape on the joint
      - the elbow pole is 0.025 away at far 0.88, so the point stops at 0.80
      - the mouth cannot open past 0.36, or the FK wrist ring 0.271 m up the forearm
        takes the centre inside its half of the summed radii
    """
    mouth, near, far = size * 0.36, size * 0.50, size * 0.80
    pin, cross, barb = size * 0.15, size * 0.24, size * 0.12
    wing, wz = size * 0.26, 0.66
    flare = far + barb * 0.55
    verts = ((0.0, 0.0, near), (0.0, mouth, far),
             (-pin, 0.0, 0.0), (pin, 0.0, 0.0),
             (-cross, 0.0, 0.0), (-cross * 1.25, mouth, 0.0),
             (cross, 0.0, 0.0), (cross * 1.25, mouth, 0.0),
             (0.0, mouth - barb, flare),
             (-wing, mouth, wz * far), (wing, mouth, wz * far))
    return verts, ((0, 1), (2, 3), (4, 5), (6, 7), (1, 8),
                   (9, 1), (10, 1), (9, 8), (10, 8))


def _wedge(size=1.0):
    """Apex on the bone head, opening down the bone to a frame. The point IS the pivot,
    which is what lets two of these share one joint and still read apart.

    Thin across bone X: on the clavicle that axis is fore-aft, and the FK arm ring lives
    out there at +-0.139. A square frame put its corners 0.006 from that ring's wire.
    """
    w, t = size * 0.45, size * 0.10
    verts = ((0.0, 0.0, 0.0), (-t, size, -w), (t, size, -w), (t, size, w), (-t, size, w))
    return verts, ((0, 1), (0, 2), (0, 3), (0, 4),
                   (1, 2), (2, 3), (3, 4), (4, 1))


def _collar(size=1.0, span=70.0, n=9):
    """FK limb ring as two arcs facing fore and aft, open along X. The rig rests with the
    arms down, so at hip height a closed ring round one thigh runs into the other thigh
    inboard and into the fist outboard; the two gaps are where those live. Symmetric, so
    its centre is still the joint and it needs no mirror."""
    half = math.radians(span) * 0.5
    r = size * 0.5
    verts, edges = [], []
    for sign in (1.0, -1.0):
        base = len(verts)
        for i in range(n):
            a = sign * (math.pi * 0.5) + (-half + 2.0 * half * i / (n - 1))
            verts.append((r * math.cos(a), 0.0, r * math.sin(a)))
        edges.extend((base + i, base + i + 1) for i in range(n - 1))
    return tuple(verts), tuple(edges)


def _belt(size=1.0, chamfer=0.35):
    """Hip ring: a chamfered square, so it clears a rectangular section at its own width.

    A circle cannot do this job. The pelvis section is 0.524 by 0.262, so a circle that
    clears its corners has to reach 0.293 out, and at that radius it runs into the fists
    hanging at the hips on the slim build. This clears the whole section at 0.275.
    """
    a = b = size * 0.5
    ix, iz = a * (1.0 - chamfer), b * (1.0 - chamfer)
    verts = ((a, 0, iz), (ix, 0, b), (-ix, 0, b), (-a, 0, iz),
             (-a, 0, -iz), (-ix, 0, -b), (ix, 0, -b), (a, 0, -iz))
    return verts, ring_loop(8)


def _ball(size=1.0, n=16):
    """Pole grip: three great circles, so it reads as something to grab from any angle.
    A flat ring is a line seen edge on, which is what made the poles hard to find."""
    r = size * 0.5
    verts, edges = [], []
    for axis in range(3):
        base = len(verts)
        for i in range(n):
            c, s = r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)
            verts.append(((0.0, c, s), (c, 0.0, s), (c, s, 0.0))[axis])
        edges.extend((base + i, base + (i + 1) % n) for i in range(n))
    return tuple(verts), tuple(edges)


def _cone(size=1.0):
    """Pole marker: a ring with a spike back along -Y, toward the joint it steers."""
    ring, edges = circle_xz(8, size * 0.5)
    verts = ring + ((0.0, -size, 0.0),)
    tip = len(verts) - 1
    return verts, tuple(edges) + tuple((i, tip) for i in range(0, 8, 2))


def _ground(size=1.0):
    """Ring in XZ with four outward arrow heads and nothing in the middle - the master
    control lies on the floor under the whole rig, so it has to stay out of the way of
    everything standing on it."""
    r = size * 0.5
    verts, edges = circle_xz(24, r)
    verts, edges = list(verts), list(edges)
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        px, pz = -dz, dx
        tip = len(verts)
        verts.extend([(dx * r * 1.24, 0.0, dz * r * 1.24),
                      ((dx * r + px * r * 0.2) * 0.98, 0.0, (dz * r + pz * r * 0.2) * 0.98),
                      ((dx * r - px * r * 0.2) * 0.98, 0.0, (dz * r - pz * r * 0.2) * 0.98)])
        edges.extend([(tip, tip + 1), (tip, tip + 2)])
    return tuple(verts), tuple(edges)


def _arrows(size=1.0):
    """Plus sign in XZ with arrow heads - the master control."""
    h = size * 0.5
    t = size * 0.12
    verts = []
    edges = []
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        base = len(verts)
        verts.extend([(0.0, 0.0, 0.0), (dx * h, 0.0, dz * h),
                      ((dx * h - dz * t) * 0.82, 0.0, (dz * h + dx * t) * 0.82),
                      ((dx * h + dz * t) * 0.82, 0.0, (dz * h - dx * t) * 0.82)])
        edges.extend([(base, base + 1), (base + 1, base + 2), (base + 1, base + 3)])
    return tuple(verts), tuple(edges)


def _make(name, data):
    return Widget(name, tuple(data[0]), tuple(data[1]))


WIDGETS = {w.name: w for w in (
    _make("circle", circle_xz(24, 0.5)),
    _make("square", _square_xz(1.0)),
    _make("cube", _box(1.0)),
    _make("diamond", _diamond(1.0)),
    _make("socket", _socket(1.0)),
    _make("wedge", _wedge(1.0)),
    _make("belt", _belt(1.0)),
    _make("collar", _collar(1.0)),
    _make("ball", _ball(1.0)),
    _make("cone", _cone(1.0)),
    _make("arrows", _arrows(1.0)),
    _make("ground", _ground(1.0)),
)}


def widget(name) -> Widget:
    return WIDGETS[name]
