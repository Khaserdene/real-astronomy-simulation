"""Headless Blender renderer for SPH gas as a glowing volume.

Imports a gas point cloud (positions), converts it to a volume with the
Geometry-Nodes "Points to Volume" node, and renders it with an emissive
Principled Volume shader in Cycles.  This is what gives gas its soft, glowing,
depth-shaded look -- the thing flat points can't do.

    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" --background \
        --python blender/render_gas.py -- \
        --ply output/gas/gas.ply --out output/gas/render.png \
        --voxel 0.35 --radius 0.7 --emission 4 --color 1.0 0.55 0.25
"""
import argparse
import os
import sys

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=os.path.abspath)
    ap.add_argument("--out", required=True, type=os.path.abspath)
    ap.add_argument("--res", type=int, default=1100)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--voxel", type=float, default=0.35, help="voxel size [kpc]")
    ap.add_argument("--radius", type=float, default=0.7, help="point radius [kpc]")
    ap.add_argument("--density", type=float, default=2.0)
    ap.add_argument("--emission", type=float, default=4.0)
    ap.add_argument("--color", type=float, nargs=3, default=[1.0, 0.55, 0.25])
    ap.add_argument("--cam-dist", type=float, default=55.0)
    ap.add_argument("--cam-elev", type=float, default=28.0)
    return ap.parse_args(argv)


def enable_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            any_on = False
            for d in prefs.devices:
                d.use = (d.type != "CPU")
                any_on = any_on or d.use
            if any_on:
                return True
        except Exception:
            pass
    return False


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # World: black.
    world = bpy.data.worlds.new("Space")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0, 0, 0, 1)
    bg.inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = world

    # Import gas points.
    bpy.ops.wm.ply_import(filepath=args.ply)
    obj = bpy.context.selected_objects[0]

    # Volume material (emissive Principled Volume).
    mat = bpy.data.materials.new("GasVolume")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (*args.color, 1.0)
    vol.inputs["Density"].default_value = args.density
    vol.inputs["Emission Strength"].default_value = args.emission
    vol.inputs["Emission Color"].default_value = (*args.color, 1.0)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])

    # Geometry nodes: points -> volume -> set material.
    ng = bpy.data.node_groups.new("PointsToVolume", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")
    p2v = ng.nodes.new("GeometryNodePointsToVolume")
    p2v.resolution_mode = "VOXEL_SIZE"
    p2v.inputs["Voxel Size"].default_value = args.voxel
    p2v.inputs["Radius"].default_value = args.radius
    setmat = ng.nodes.new("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = mat
    ng.links.new(gin.outputs[0], p2v.inputs["Points"])
    ng.links.new(p2v.outputs["Volume"], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs["Geometry"], gout.inputs[0])
    obj.modifiers.new("ToVolume", "NODES").node_group = ng

    # Camera.
    import math
    target = bpy.data.objects.new("T", None)
    bpy.context.collection.objects.link(target)
    cam_data = bpy.data.cameras.new("Cam"); cam_data.clip_end = 1e5
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    el = math.radians(args.cam_elev)
    cam.location = (0, -args.cam_dist * math.cos(el), args.cam_dist * math.sin(el))
    c = cam.constraints.new("TRACK_TO"); c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    bpy.context.scene.camera = cam

    # Render (Cycles GPU, glare bloom).
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x = sc.render.resolution_y = args.res
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "Standard"
    sc.cycles.samples = args.samples
    if enable_gpu():
        sc.cycles.device = "GPU"
    sc.use_nodes = True
    ct = sc.node_tree; ct.nodes.clear()
    rl = ct.nodes.new("CompositorNodeRLayers")
    glare = ct.nodes.new("CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"; glare.size = 6
    comp = ct.nodes.new("CompositorNodeComposite")
    ct.links.new(rl.outputs["Image"], glare.inputs["Image"])
    ct.links.new(glare.outputs["Image"], comp.inputs["Image"])

    sc.render.filepath = args.out
    bpy.ops.render.render(write_still=True)
    print(f"[gas] rendered -> {args.out}")


if __name__ == "__main__":
    main()
