"""Walks an efb.rigdef.RigDefinition and produces a real armature object."""

import math

import bpy

from efb.rigdef import ADDON_VERSION, COLOURS, RIG_ID, DistanceVar, limbs_to_json

from .animscene import from_matrix, to_matrix

SPAWN_YAW = math.pi


def widget_object(w, prefix="WGT-", collection=None):
    name = prefix + w.name
    obj = bpy.data.objects.get(name)
    if obj is not None:
        return obj
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in w.verts], [tuple(e) for e in w.edges], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    if collection is not None:
        collection.objects.link(obj)
    return obj


def build_rig(rigdef, name=None, context=None):
    """Creates the armature object. Returns it, active and in object mode."""
    ctx = context or bpy.context
    name = name or rigdef.name

    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    ctx.collection.objects.link(obj)
    ctx.view_layer.objects.active = obj
    obj.select_set(True)
    data["rig_id"] = RIG_ID
    data["rig_version"] = rigdef.version
    data["rig_addon"] = ADDON_VERSION
    data["efb_limbs"] = limbs_to_json(rigdef.limbs)

    _edit_bones(obj, data, rigdef)
    _collections(data, rigdef)
    _properties(obj, rigdef)
    _pose(obj, data, rigdef, ctx)
    _constraints(obj, rigdef)
    _seat_child_of(obj, rigdef, ctx)
    _drivers(obj, rigdef)
    _flag_drivers(obj, data, rigdef)
    _collection_drivers(obj, data, rigdef)

    data.display_type = "OCTAHEDRAL"
    data.show_bone_custom_shapes = True
    obj.show_in_front = True
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, SPAWN_YAW)
    ctx.view_layer.update()
    return obj


def _edit_bones(obj, data, rigdef):
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for b in rigdef.bones:
            eb = data.edit_bones.new(b.name)
            if eb.name != b.name:
                raise RuntimeError("bone name collision on %r" % b.name)
            eb.head = (0.0, 0.0, 0.0)
            eb.tail = (0.0, b.length, 0.0)
            eb.matrix = to_matrix(b.rest)
            eb.use_deform = b.use_deform
            eb.use_connect = False
        for b in rigdef.bones:
            if b.parent:
                data.edit_bones[b.name].parent = data.edit_bones[b.parent]
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def _collections(data, rigdef):
    for c in rigdef.collections:
        coll = data.collections.new(c.name)
        coll.is_visible = c.visible
    for b in rigdef.bones:
        data.collections[b.collection].assign(data.bones[b.name])


def _properties(obj, rigdef):
    for p in rigdef.properties:
        if p.kind == "bool":
            obj[p.name] = bool(p.default)
        elif p.kind == "int":
            obj[p.name] = int(p.default)
        else:
            obj[p.name] = float(p.default)
        ui = obj.id_properties_ui(p.name)
        if p.kind == "bool":
            ui.update(default=bool(p.default), description=p.description)
        elif p.kind == "int":
            ui.update(min=int(p.minimum), max=int(p.maximum), soft_min=int(p.minimum),
                      soft_max=int(p.maximum), default=int(p.default),
                      description=p.description)
        else:
            ui.update(min=p.minimum, max=p.maximum, soft_min=p.minimum,
                      soft_max=p.maximum, default=float(p.default),
                      description=p.description)


def _pose(obj, data, rigdef, ctx):
    wgt = bpy.data.collections.new("WGT-" + obj.name)
    ctx.scene.collection.children.link(wgt)
    wgt.hide_viewport = True
    wgt.hide_render = True
    shapes = {k: widget_object(w, collection=wgt) for k, w in rigdef.widgets.items()}

    for b in rigdef.bones:
        pb = obj.pose.bones[b.name]
        bone = data.bones[b.name]
        pb.rotation_mode = b.rotation_mode
        if b.lock_location:
            pb.lock_location = (True, True, True)
        bone.hide = b.hide
        bone.hide_select = b.lock_select
        if b.ik_lock:
            pb.lock_ik_x, pb.lock_ik_y, pb.lock_ik_z = b.ik_lock
        if b.ik_limit:
            pb.use_ik_limit_x = True
            pb.ik_min_x = math.radians(b.ik_limit[0])
            pb.ik_max_x = math.radians(b.ik_limit[1])
        palette = COLOURS.get(b.collection, "DEFAULT")
        if palette != "DEFAULT":
            bone.color.palette = palette
        if b.widget:
            pb.custom_shape = shapes[b.widget]
            pb.use_custom_shape_bone_size = False
            s = b.widget_size
            pb.custom_shape_scale_xyz = (s, s, s)
            pb.custom_shape_translation = b.widget_translation
            pb.custom_shape_rotation_euler = tuple(math.radians(a)
                                                   for a in b.widget_rotation)


def _constraints(obj, rigdef):
    for c in rigdef.constraints:
        pb = obj.pose.bones[c.owner]
        con = pb.constraints.new(c.type)
        con.name = c.name
        con.target = obj
        con.subtarget = c.subtarget
        con.influence = c.influence
        if c.type == "IK":
            con.chain_count = c.chain_count
            con.use_rotation = c.use_rotation
            con.use_stretch = c.use_stretch
            if c.pole_subtarget:
                con.pole_target = obj
                con.pole_subtarget = c.pole_subtarget
                con.pole_angle = c.pole_angle
        elif c.type == "DAMPED_TRACK":
            con.track_axis = c.track_axis
        elif c.type == "LOCKED_TRACK":
            con.track_axis = c.track_axis
            con.lock_axis = c.lock_axis
        elif c.type == "LIMIT_DISTANCE":
            con.distance = c.distance
            con.limit_mode = c.limit_mode
        elif c.type in ("COPY_SCALE", "COPY_ROTATION"):
            if c.use_axis:
                con.use_x, con.use_y, con.use_z = c.use_axis
        if c.target_space:
            con.target_space = c.target_space
        if c.owner_space:
            con.owner_space = c.owner_space


def _seat_child_of(obj, rigdef, ctx):
    """A Child Of has to be handed the inverse of its target's rest pose, or the owner is
    offset by the target's whole transform the moment the constraint is added. Measured
    against the alternatives on a probe rig: the identity throws the owner three metres,
    the target's pose matrix inverted holds it to 0.000000, and it keeps holding when the
    object itself is yawed.
    """
    ctx.view_layer.update()
    for c in rigdef.constraints:
        if not c.seat_inverse:
            continue
        con = obj.pose.bones[c.owner].constraints[c.name]
        con.inverse_matrix = obj.pose.bones[c.subtarget].matrix.inverted()


def _driver(owner, obj, spec):
    index = getattr(spec, "index", -1)
    fcurve = (owner.driver_add(spec.data_path, index) if index >= 0
              else owner.driver_add(spec.data_path))
    for mod in list(fcurve.modifiers):
        fcurve.modifiers.remove(mod)
    drv = fcurve.driver
    drv.type = "SCRIPTED"
    for existing in list(drv.variables):
        drv.variables.remove(existing)
    for v in spec.variables:
        var = drv.variables.new()
        var.name = v.name
        if isinstance(v, DistanceVar):
            var.type = "LOC_DIFF"
            for slot, bone in zip(var.targets, (v.bone_a, v.bone_b)):
                slot.id = obj
                slot.bone_target = bone
            continue
        var.type = "SINGLE_PROP"
        var.targets[0].id_type = "OBJECT"
        var.targets[0].id = obj
        var.targets[0].data_path = '["%s"]' % v.prop
    drv.expression = spec.expression


def _drivers(obj, rigdef):
    for d in rigdef.drivers:
        _driver(obj, obj, d)


def _flag_drivers(obj, data, rigdef):
    for d in rigdef.flag_drivers:
        _driver(data if d.on_data else obj, obj, d)


def _collection_drivers(obj, data, rigdef):
    """Visibility lives on the armature data, so these hang off it rather than the object.
    They are what keeps an inert control from being drawn."""
    for d in rigdef.collection_drivers:
        _driver(data, obj, d)
