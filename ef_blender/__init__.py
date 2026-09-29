"""Extension entry point. The whole addon is two packages sitting next to this file:

  efb     pure python, never imports bpy - matrices, file formats, rig definition
  efbpy   the bpy shell - operators, panels, scene reading and writing

Blender imports this module as bl_ext.<repo>.ef_blender, but efbpy's modules import efb
by its plain name so the same tree runs under python3 with no Blender at all. Putting
this directory on sys.path is what makes both spellings resolve to one copy.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import efbpy


def register():
    efbpy.register()


def unregister():
    efbpy.unregister()
