"""The Epic Fight body mesh, built next to the rig and bound to it.

The json's positions are already in the armature's rest space, so the object stays at the
origin and an Armature modifier plus one vertex group per joint is the whole bind. The
default skin is the vanilla texture that ships in efb/data; it is packed into the .blend
so the file survives without the addon.
"""

import os

import bpy

from efb import bundled

__all__ = ["build_body", "bind", "make_skin_material", "load_skin_image",
           "skin_image_of", "bodies_of", "skin_of", "set_skin", "BODY_SUFFIX"]

BODY_SUFFIX = "-body"


def _same_file(a, b):
    if not a or not b:
        return False
    return (os.path.normcase(os.path.abspath(bpy.path.abspath(a)))
            == os.path.normcase(os.path.abspath(bpy.path.abspath(b))))


def load_skin_image(path):
    """Reuse by path, not by name - wide and slim are both 64x64 pngs in one folder."""
    img = next((i for i in bpy.data.images if _same_file(i.filepath, path)), None)
    if img is None:
        img = bpy.data.images.load(path)
        img.name = os.path.basename(path)
    try:
        img.pack()
    except RuntimeError:
        pass
    return img


def skin_image_of(material):
    if not material or not material.use_nodes:
        return None
    return next((n.image for n in material.node_tree.nodes
                 if n.type == "TEX_IMAGE" and n.image), None)


def make_skin_material(skin_path, name=None):
    name = name or ("EF-skin." + (os.path.splitext(os.path.basename(skin_path))[0]
                                  if skin_path else "plain"))
    m = bpy.data.materials.get(name)
    if m is not None:
        img = skin_image_of(m)
        if _same_file(getattr(img, "filepath", ""), skin_path) or not skin_path:
            return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.8
    if skin_path:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = load_skin_image(skin_path)
        tex.interpolation = "Closest"
        tex.location = (-620, 0)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        cut = nt.nodes.new("ShaderNodeMath")
        cut.operation = "GREATER_THAN"
        cut.inputs[1].default_value = 0.5
        cut.label = cut.name = "AlphaClip"
        cut.location = (-320, -220)
        nt.links.new(tex.outputs["Alpha"], cut.inputs[0])
        nt.links.new(cut.outputs["Value"], bsdf.inputs["Alpha"])
    m.alpha_threshold = 0.5
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "DITHERED"
    else:
        m.blend_method = "CLIP"
    m.use_backface_culling = False
    return m


def bodies_of(rig):
    """Both of one rig's bodies: the game body and the authoring copy, in that order.

    Only this rig's. A scene holds several and reskinning one must not reach the others.
    """
    from . import smooth
    return [o for o in (smooth.body_of(rig), smooth.smooth_body_of(rig)) if o is not None]


def skin_of(rig):
    """The image both bodies wear, or None."""
    for obj in bodies_of(rig):
        img = skin_image_of(obj.data.materials[0] if obj.data.materials else None)
        if img is not None:
            return img
    return None


def set_skin(rig, skin_path=None, context=None):
    """Retargets both of `rig`'s bodies to a skin, keeping everything else.

    Nothing here touches the mesh, the bind or a pose bone, so the pose and the action
    survive - that is the whole point of it over regenerating. A blank path is the
    bundled skin for the build the body was made from. The material is assigned, never
    edited in place: two rigs generated with the same skin share one, and mutating it
    would repaint both.
    """
    ctx = context or bpy.context
    bodies = bodies_of(rig)
    if not bodies:
        raise RuntimeError("%s carries no body to skin" % rig.name)
    variant = bodies[0].get("efb_body") or bundled.DEFAULT_VARIANT
    path = skin_path or bundled.skin_path(variant)
    material = make_skin_material(path)
    for obj in bodies:
        me = obj.data
        me.materials.clear()
        me.materials.append(material)
        me.update()
    rig.update_tag()
    ctx.view_layer.update()
    return material, bodies


def build_body(mesh, rig=None, name=None, skin_path=None, variant=None, context=None):
    """Creates the mesh object, binds it to `rig` and returns it.

    `mesh` is an efb.meshjson.BodyMesh. `skin_path` defaults to the bundled skin for
    `variant`; pass "" for an untextured body.
    """
    ctx = context or bpy.context
    name = name or ((rig.name if rig else "EF") + BODY_SUFFIX)
    if skin_path is None:
        skin_path = bundled.skin_path(variant or bundled.DEFAULT_VARIANT)

    faces, loop_uv, loop_n, _parts = mesh.loops()
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in mesh.positions], [], faces)
    me.update()
    if len(me.loops) != len(loop_uv):
        raise RuntimeError("%s: %d loops built from %d corners"
                           % (name, len(me.loops), len(loop_uv)))
    _uvs(me, loop_uv)
    _normals(me, loop_n)

    obj = bpy.data.objects.new(name, me)
    ctx.collection.objects.link(obj)
    for jname, pairs in mesh.group_weights().items():
        g = obj.vertex_groups.new(name=jname)
        for vi, weight in pairs:
            g.add([vi], float(weight), "REPLACE")
    if skin_path:
        me.materials.append(make_skin_material(skin_path))

    obj["efb_body"] = variant or bundled.DEFAULT_VARIANT
    obj["efb_space"] = "armature_rest"
    if rig is not None:
        bind(obj, rig)
    return obj


def bind(obj, rig):
    """Parent to the rig and add the Armature modifier. Both sit at the origin, so the
    parent inverse stays identity and the raw rest positions line up."""
    obj.parent = rig
    obj.matrix_parent_inverse.identity()
    obj.matrix_basis.identity()
    mod = obj.modifiers.get("Armature") or obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    mod.use_vertex_groups = True
    return mod


def _uvs(me, loop_uv):
    """Epic Fight runs V from the top of the sheet, Blender reads it from the bottom."""
    layer = me.uv_layers.new(name="UVMap")
    flat = []
    for u, v in loop_uv:
        flat.append(u)
        flat.append(1.0 - v)
    layer.uv.foreach_set("vector", flat)
    me.update()


def _normals(me, loop_n):
    try:
        me.normals_split_custom_set([tuple(n) for n in loop_n])
    except (RuntimeError, TypeError):
        pass
    me.update()
