"""Blender <-> Minecraft basis, derived from Epic Fight's loaders.

Established at source (epicfight-21.17.3.1.jar):

  JsonAssetLoader.BLENDER_TO_MINECRAFT_COORD = OpenMatrix4f.createRotatorDeg(-90, X_AXIS)

which expands to the row-major matrix below, i.e. (x, y, z)_blender -> (x, z, -y)_minecraft.

Where Epic Fight applies it:
  * mesh positions and normals   - every vertex, unconditionally
  * armature json, root joint    - localMatrix.mulFront(B), root only
  * animation json, first track  - keyframe.mulFront(B), MATRIX format only, and only
                                   for the first track that names a known joint or "Coord"

For the root joint the two cancel: the clip keyframe becomes
inv(B @ L_root) @ B @ A_root == inv(L_root) @ A_root, the same expression every other
joint gets. So an exported animation is pure Blender data end to end and needs no basis
change; B only shows up when you want to predict the in-game model-space matrix.
"""

from __future__ import annotations

from .matrix import Mat4

__all__ = [
    "BLENDER_TO_MINECRAFT", "MINECRAFT_TO_BLENDER",
    "blender_to_minecraft", "minecraft_to_blender",
    "rebase_blender_to_minecraft", "rebase_minecraft_to_blender",
    "ef_root_local", "ef_root_local_to_blender",
    "ef_delta_from_pose", "ef_pose_from_delta",
]

BLENDER_TO_MINECRAFT = Mat4((1.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, -1.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 1.0))

MINECRAFT_TO_BLENDER = Mat4((1.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, -1.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 1.0))


def blender_to_minecraft(p):
    """(x, y, z) -> (x, z, -y)"""
    x, y, z = p
    return (x, z, -y)


def minecraft_to_blender(p):
    """(x, y, z) -> (x, -z, y)"""
    x, y, z = p
    return (x, -z, y)


def rebase_blender_to_minecraft(m: Mat4) -> Mat4:
    """Re-express a whole transform in the other basis: B @ m @ B^-1.
    Use for a transform whose input and output both live in world space."""
    return BLENDER_TO_MINECRAFT @ m @ MINECRAFT_TO_BLENDER


def rebase_minecraft_to_blender(m: Mat4) -> Mat4:
    return MINECRAFT_TO_BLENDER @ m @ BLENDER_TO_MINECRAFT


def ef_root_local(rest_local: Mat4) -> Mat4:
    """Joint.localTransform as Epic Fight holds it for the ROOT joint: B @ L.
    One-sided on purpose - the root's parent is the Minecraft world, its own frame
    stays the Blender bone frame, so every child local stays untouched."""
    return BLENDER_TO_MINECRAFT @ rest_local


def ef_root_local_to_blender(ef_local: Mat4) -> Mat4:
    """Inverse of ef_root_local; what Armature.exportJoint writes back out."""
    return MINECRAFT_TO_BLENDER @ ef_local


def ef_delta_from_pose(rest_local: Mat4, parent_relative_pose: Mat4) -> Mat4:
    """inv(L) @ A - the value Epic Fight keeps in a TransformSheet keyframe, and the
    same thing Blender calls pose_bone.matrix_basis."""
    return rest_local.inverse() @ parent_relative_pose


def ef_pose_from_delta(rest_local: Mat4, delta: Mat4) -> Mat4:
    """L @ delta - the parent-relative pose matrix a MATRIX-format json stores."""
    return rest_local @ delta
