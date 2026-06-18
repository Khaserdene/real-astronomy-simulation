"""Astronomy Sim Importer -- Blender 4.2+ extension.

Adds an "Astro Sim" panel to the 3D viewport's N-sidebar: point it at a
simulation output folder (the one full of ``snap_*.h5``) and press Import to load
the whole run as a Blender animation -- press Spacebar to play it in the viewport,
scrub the timeline, art-direct the materials/lighting/camera, and render.

The heavy lifting lives in :mod:`loader` (snapshot reading + per-frame geometry);
this file is just the Blender UI/registration.
"""
from __future__ import annotations

import bpy
from bpy.props import StringProperty
from bpy.types import Operator, Panel

from . import loader


class ASTRO_OT_import_folder(Operator):
    bl_idname = "astro.import_folder"
    bl_label = "Import simulation folder"
    bl_description = "Load all snap_*.h5 in the folder as an animation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        folder = context.scene.astro_sim_folder
        if not folder:
            self.report({"ERROR"}, "Choose a simulation folder first")
            return {"CANCELLED"}
        try:
            n = loader.load_folder(bpy.path.abspath(folder))
        except FileNotFoundError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        except ImportError:
            self.report({"ERROR"},
                        "h5py is not available to Blender. Bundle the h5py wheel "
                        "in the extension (see README) or install it.")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Loaded {n} frames -- press Spacebar to play")
        return {"FINISHED"}


class ASTRO_PT_panel(Panel):
    bl_label = "Astro Sim"
    bl_idname = "ASTRO_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Astro Sim"

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(context.scene, "astro_sim_folder", text="")
        col.operator("astro.import_folder", icon="IMPORT")
        if loader._STATE["snaps"]:
            col.separator()
            col.label(text=f"{len(loader._STATE['snaps'])} frames loaded")


_CLASSES = (ASTRO_OT_import_folder, ASTRO_PT_panel)


def register():
    bpy.types.Scene.astro_sim_folder = StringProperty(
        name="Simulation folder", subtype="DIR_PATH",
        description="Folder containing snap_*.h5 snapshots")
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.astro_sim_folder


if __name__ == "__main__":
    register()
