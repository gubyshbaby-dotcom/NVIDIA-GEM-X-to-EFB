"""Epic Fight's own skinning, so a Blender body can be checked against it.

Both of the mod's paths do the same thing and nothing else. SkinnedMesh.getVertexPosition
on the CPU and assets/epicfight/shaders/compute/vanilla_mesh_transformer.comp on the GPU
both accumulate `fpos += (poses[j] * pos) * weight` - plain linear blend skinning, no
renormalisation, no dual quaternion, nothing between the joint matrices and the vertex.
The GPU adds a per-mesh-part matrix in front of the joint matrix, identity for the body.

The mesh json stores Blender coordinates and JsonAssetLoader multiplies every position by
B at load, so the whole of it is

    v' = sum_j w_j * (Model_j @ toOrigin_j) @ (B @ v)

with Model_j = Model_parent @ localTransform_j @ delta_j, which efb.armature already
walks. This file is the vertex half and nothing more.
"""

from __future__ import annotations

from .coords import BLENDER_TO_MINECRAFT, MINECRAFT_TO_BLENDER

__all__ = ["skin_minecraft", "skin_blender", "rest_error", "influence_census",
           "rigid_owner", "distances"]


def skin_minecraft(arm, mesh, deltas=None):
    """Every vertex where the game would put it, Minecraft space.

    `deltas` maps joint name -> local delta, the same value a TransformSheet keyframe
    holds and the same one Blender's pose bone carries against its rest. Missing joints
    are at rest.
    """
    skinning = arm.skinning_matrices(deltas or {})
    out = []
    for position, influences in zip(mesh.positions, mesh.skin):
        p = BLENDER_TO_MINECRAFT.transform_point(position)
        x = y = z = 0.0
        for jname, weight in influences:
            qx, qy, qz = skinning[jname].transform_point(p)
            x += qx * weight
            y += qy * weight
            z += qz * weight
        out.append((x, y, z))
    return out


def skin_blender(arm, mesh, deltas=None):
    """The same answer in Blender space, which is what a mesh can be compared against."""
    return [MINECRAFT_TO_BLENDER.transform_point(p)
            for p in skin_minecraft(arm, mesh, deltas)]


def rest_error(arm, mesh) -> float:
    """Worst distance between the rest mesh and this pipeline's answer for no pose.

    The known-answer check: the coordinate handling and the toOrigin bake are only right
    if it comes out at machine epsilon.
    """
    return max(distances(mesh.positions, skin_blender(arm, mesh)))


def influence_census(mesh) -> dict:
    """influence count -> how many vertices carry that many."""
    out = {}
    for infl in mesh.skin:
        out[len(infl)] = out.get(len(infl), 0) + 1
    return out


def rigid_owner(mesh, index):
    """The one joint a vertex belongs to at weight 1.0, or None if it is blended."""
    infl = mesh.skin[index]
    if len(infl) == 1 and abs(infl[0][1] - 1.0) < 1e-6:
        return infl[0][0]
    return None


def distances(a, b):
    return [sum((p - q) ** 2 for p, q in zip(u, v)) ** 0.5 for u, v in zip(a, b)]
