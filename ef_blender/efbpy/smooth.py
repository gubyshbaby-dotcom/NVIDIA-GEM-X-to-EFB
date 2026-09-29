"""The stylised body: the same mesh, cut into loops along each limb and weighted across
the joint so the limb bends as one piece.

Two objects, one rig. Epic Fight's own body keeps its 260 vertices and its single
influences and reproduces the game's skinning to 1e-6 m; this one is a copy that is only
ever drawn, and it is 0.043 to 0.196 m off the game on shipped poses. Which of the two is
visible is driven off one property on the rig, so the switch is live, keyable and cannot
get out of step with what the panel says.
"""

import bmesh
import bpy

from efb.rigdef import SMOOTH_DEFORM, SUBDIVISION
from efb.smoothdef import SMOOTH_LIMBS, SMOOTH_SUFFIX, blend_band, lower_share

from .body import bind

__all__ = ["build_smooth_body", "smooth_body_of", "body_of", "is_smooth", "set_smooth",
           "SMOOTH_SUFFIX"]

SMOOTH_KEY = "efb_smooth_body"

LOOP_MERGE = 0.005


def body_of(rig):
    return next((o for o in bpy.data.objects
                 if o.parent is rig and o.type == "MESH" and o.get("efb_body")), None)


def smooth_body_of(rig):
    return next((o for o in bpy.data.objects
                 if o.parent is rig and o.type == "MESH" and o.get(SMOOTH_KEY)), None)


def is_smooth(rig) -> bool:
    return bool(rig.get(SMOOTH_DEFORM))


def set_smooth(rig, on, context=None):
    rig[SMOOTH_DEFORM] = bool(on)
    rig.update_tag()
    (context or bpy.context).view_layer.update()


def build_smooth_body(body, rig, context=None):
    """Creates the authoring body next to `body` and wires the visibility switch."""
    ctx = context or bpy.context
    old = smooth_body_of(rig)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)

    name = body.name + SMOOTH_SUFFIX
    me = body.data.copy()
    me.name = name
    obj = bpy.data.objects.new(name, me)
    ctx.collection.objects.link(obj)
    for g in body.vertex_groups:
        obj.vertex_groups.new(name=g.name)

    for limb in SMOOTH_LIMBS:
        _resection(obj, rig, limb)

    obj[SMOOTH_KEY] = 1
    obj["efb_space"] = "armature_rest"
    bind(obj, rig).use_deform_preserve_volume = True
    _visibility_drivers(body, obj, rig)
    return obj


def _limb_frame(rig, limb):
    """(joint, axis, band) in armature space. The axis is the upper bone's own: the wrist
    hangs 0.03 back, so the shoulder-to-wrist line would tilt every cut by 3 degrees."""
    bones = rig.data.bones
    upper, lower = bones[limb.upper], bones[limb.lower]
    joint = lower.head_local.copy()
    axis = (joint - upper.head_local).normalized()
    return joint, axis, blend_band(upper.length, lower.length)


def _island(bm, layer, mine):
    """The limb, read off the weights: a cut renumbers the mesh, but bmesh interpolates the
    deform layer onto new vertices, so a new loop carries the limb's groups by itself."""
    verts = [v for v in bm.verts if any(g in v[layer] for g in mine)]
    inside = set(verts)
    faces = [f for f in bm.faces if all(v in inside for v in f.verts)]
    edges = [e for e in bm.edges if all(v in inside for v in e.verts)]
    return verts, verts + edges + faces


def _resection(obj, rig, limb):
    joint, axis, band = _limb_frame(rig, limb)
    groups = obj.vertex_groups
    mine = {groups[n].index for n in (limb.upper, limb.lower, limb.marker)
            if n in groups}
    if not mine:
        return
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    layer = bm.verts.layers.deform.active or bm.verts.layers.deform.new()

    verts, geom = _island(bm, layer, mine)
    if not verts:
        bm.free()
        return
    along = sorted(v.co.dot(axis) for v in verts)
    lo, hi = along[0], along[-1]
    for i in range(1, SUBDIVISION):
        station = lo + (hi - lo) * i / SUBDIVISION
        if min(abs(station - a) for a in along) < LOOP_MERGE:
            continue
        bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6,
                               plane_co=axis * station, plane_no=axis,
                               clear_inner=False, clear_outer=False)
        _verts, geom = _island(bm, layer, mine)

    _reweight(bm, layer, groups, limb, joint, axis, band, mine)
    bm.to_mesh(me)
    bm.free()
    me.update()


def _stations(values):
    """Rest positions grouped into the loops they form. The hairline is three rings 0.0008
    apart and one shared blend value is what shuts it exactly rather than to a mm."""
    out = []
    for value in sorted(values):
        if out and value - out[-1][-1] <= LOOP_MERGE:
            out[-1].append(value)
        else:
            out.append([value])
    return {v: sum(g) / len(g) for g in out for v in g}


def _reweight(bm, layer, groups, limb, joint, axis, band, mine):
    """Every limb vertex becomes a blend of the two bones and nothing else. The seam marker
    stops deforming: the blend closes the seam, and its slide would open it again."""
    up, low = groups[limb.upper].index, groups[limb.lower].index
    mine_verts = [v for v in bm.verts if any(g in v[layer] for g in mine)]
    station = _stations({v.co.dot(axis) for v in mine_verts})
    base = joint.dot(axis)
    for v in mine_verts:
        d = v[layer]
        share = lower_share(station[v.co.dot(axis)] - base, band)
        for g in list(d.keys()):
            if g in mine:
                del d[g]
        d[up] = 1.0 - share
        d[low] = share


def _visibility_drivers(body, smooth, rig):
    """One property, two objects. Both hide_viewport and hide_render, so a render cannot
    quietly come out of the wrong body."""
    for obj, expression in ((body, "s"), (smooth, "1 - s")):
        for path in ("hide_viewport", "hide_render"):
            obj.driver_remove(path)
            fcurve = obj.driver_add(path)
            drv = fcurve.driver
            drv.type = "SCRIPTED"
            for existing in list(drv.variables):
                drv.variables.remove(existing)
            var = drv.variables.new()
            var.name = "s"
            var.type = "SINGLE_PROP"
            var.targets[0].id_type = "OBJECT"
            var.targets[0].id = rig
            var.targets[0].data_path = '["%s"]' % SMOOTH_DEFORM
            drv.expression = expression
