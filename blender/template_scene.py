"""Interactive Blender handoff: load a frame as art-directable objects.

Launched (normally NOT with --background) by the GUI's "Open in Blender" button:

    blender --python blender/template_scene.py -- <frame_dir>

``<frame_dir>`` holds the per-type PLYs from ``export.to_blender`` (stars.ply,
gas.ply, dm.ply).  This builds a starter scene -- one object + named material per
type (Stars = emissive points coloured by blackbody; Gas = Principled Volume;
DarkMatter = faint points), plus camera, black world and a glare in the
compositor -- then leaves Blender open so you can tweak materials, lighting and
camera and render from the UI.  Run with --background and it just builds and
exits (used for testing).
"""
import os
import sys

import bpy


def _argv_dir():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv:
        raise SystemExit("usage: blender --python template_scene.py -- <frame_dir>")
    return os.path.abspath(argv[0])


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups):
        for item in list(coll):
            coll.remove(item)


def setup_world_camera():
    world = bpy.data.worlds.new("Space")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0, 0, 0, 1)
    bg.inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = world

    import math
    target = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(target)
    cam_data = bpy.data.cameras.new("Cam"); cam_data.clip_end = 1e5
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    el = math.radians(28)
    cam.location = (0, -60 * math.cos(el), 60 * math.sin(el))
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam


def _import(path):
    bpy.ops.wm.ply_import(filepath=path)
    return bpy.context.selected_objects[0]


def emissive_points_material(name, color_attr=None, color=(1, 1, 1), strength=1.3):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if color_attr:
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = color_attr
        nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    else:
        emit.inputs["Color"].default_value = (*color, 1.0)
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def points_modifier(obj, radius, mat):
    ng = bpy.data.node_groups.new(obj.name + "_pts", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput"); gout = ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    m2p.inputs["Radius"].default_value = radius
    sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = mat
    ng.links.new(gin.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(m2p.outputs["Points"], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], gout.inputs[0])
    obj.modifiers.new("Points", "NODES").node_group = ng


def volume_modifier(obj, voxel, radius, mat):
    ng = bpy.data.node_groups.new(obj.name + "_vol", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput"); gout = ng.nodes.new("NodeGroupOutput")
    p2v = ng.nodes.new("GeometryNodePointsToVolume")
    # Voxel-size mode: enum property on Blender <=4.x, menu input on 5.x.
    if hasattr(p2v, "resolution_mode"):
        p2v.resolution_mode = "VOXEL_SIZE"
    elif "Resolution Mode" in p2v.inputs:
        p2v.inputs["Resolution Mode"].default_value = "Size"
    p2v.inputs["Voxel Size"].default_value = voxel
    p2v.inputs["Radius"].default_value = radius
    sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = mat
    ng.links.new(gin.outputs[0], p2v.inputs["Points"])
    ng.links.new(p2v.outputs["Volume"], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], gout.inputs[0])
    obj.modifiers.new("Volume", "NODES").node_group = ng


def gas_volume_material():
    mat = bpy.data.materials.new("Gas")
    mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (1.0, 0.55, 0.25, 1.0)
    vol.inputs["Density"].default_value = 0.15
    vol.inputs["Emission Strength"].default_value = 0.8
    vol.inputs["Emission Color"].default_value = (1.0, 0.55, 0.25, 1.0)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    return mat


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.view_settings.view_transform = "Standard"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA"):
            prefs.compute_device_type = backend
            prefs.get_devices()
            if any(d.type != "CPU" for d in prefs.devices):
                for d in prefs.devices:
                    d.use = (d.type != "CPU")
                sc.cycles.device = "GPU"
                break
    except Exception:
        pass
    sc.use_nodes = True
    nt = sc.node_tree; nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    glare = nt.nodes.new("CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"; glare.size = 6
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], glare.inputs["Image"])
    nt.links.new(glare.outputs["Image"], comp.inputs["Image"])


def main():
    folder = _argv_dir()
    clear_scene()
    setup_world_camera()

    stars = os.path.join(folder, "stars.ply")
    gas = os.path.join(folder, "gas.ply")
    dm = os.path.join(folder, "dm.ply")

    if os.path.exists(stars):
        obj = _import(stars); obj.name = "Stars"
        points_modifier(obj, 0.08,
                        emissive_points_material("Stars", color_attr="Col",
                                                 strength=1.4))
    if os.path.exists(gas):
        obj = _import(gas); obj.name = "Gas"
        volume_modifier(obj, 0.3, 0.6, gas_volume_material())
    if os.path.exists(dm):
        obj = _import(dm); obj.name = "DarkMatter"
        points_modifier(obj, 0.05,
                        emissive_points_material("DarkMatter",
                                                 color=(0.25, 0.3, 0.45),
                                                 strength=0.3))
    setup_render()
    print(f"[template] scene ready from {folder} "
          f"({'stars ' if os.path.exists(stars) else ''}"
          f"{'gas ' if os.path.exists(gas) else ''}"
          f"{'dm' if os.path.exists(dm) else ''})")


if __name__ == "__main__":
    main()
