"""Headless Blender batch renderer for galaxy point-cloud frames.

Run with Blender (not the venv):

    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" --background \
        --python blender/render_frames.py -- \
        --ply output/run_high/ply --out output/run_high/render \
        --samples 48 --res 1280 --radius 0.12 --emission 6

Each PLY (a coloured star point cloud exported by export.to_pointcloud) is
imported, turned into emissive points via Geometry Nodes, and rendered with
Cycles on the GPU.  A Fog-Glow glare node in the compositor gives the soft
galactic bloom.  Output is a PNG per frame, ready to assemble into a video.
"""
import argparse
import glob
import os
import sys

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=os.path.abspath,
                    help="folder of *.ply frames")
    ap.add_argument("--out", required=True, type=os.path.abspath,
                    help="output folder for PNGs")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--res", type=int, default=1280)
    ap.add_argument("--radius", type=float, default=0.09, help="point radius [kpc]")
    ap.add_argument("--emission", type=float, default=1.3)
    ap.add_argument("--glare", type=int, default=5, help="fog-glow size (0=off)")
    ap.add_argument("--cam-dist", type=float, default=55.0)
    ap.add_argument("--cam-elev", type=float, default=28.0, help="degrees")
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT"])
    return ap.parse_args(argv)


def enable_gpu():
    """Turn on GPU compute for Cycles (OPTIX preferred, else CUDA)."""
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
                print(f"[render] GPU backend: {backend}")
                return True
        except Exception as e:
            print(f"[render] {backend} unavailable: {e}")
    print("[render] falling back to CPU")
    return False


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups):
        for item in list(block):
            block.remove(item)


def make_star_material(emission: float) -> bpy.types.Material:
    mat = bpy.data.materials.new("StarMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"          # PLY vertex colour attribute
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = emission
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_points_modifier_group(radius: float, mat) -> bpy.types.NodeTree:
    """Geometry-nodes group: mesh vertices -> emissive points."""
    ng = bpy.data.node_groups.new("StarsToPoints", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    m2p.inputs["Radius"].default_value = radius
    setmat = ng.nodes.new("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = mat
    ng.links.new(gin.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(m2p.outputs["Points"], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs["Geometry"], gout.inputs[0])
    return ng


def setup_world():
    world = bpy.data.worlds.new("Space")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = world


def setup_camera(dist: float, elev_deg: float):
    import math
    target = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(target)
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.clip_end = 100000.0
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    elev = math.radians(elev_deg)
    cam.location = (0.0, -dist * math.cos(elev), dist * math.sin(elev))
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam


def setup_compositor_glare(size: int):
    scene = bpy.context.scene
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    if size <= 0:
        nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
        return
    glare = nt.nodes.new("CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"
    glare.quality = "HIGH"
    glare.size = size
    nt.links.new(rl.outputs["Image"], glare.inputs["Image"])
    nt.links.new(glare.outputs["Image"], comp.inputs["Image"])


def setup_render(args):
    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        if enable_gpu():
            scene.cycles.device = "GPU"


def color_attr_name(obj) -> str:
    ca = obj.data.color_attributes
    return ca[0].name if len(ca) else "Col"


def render_frame(ply_path, out_png, ng_factory, mat):
    bpy.ops.wm.ply_import(filepath=ply_path)
    obj = bpy.context.selected_objects[0]
    # Point the material's Attribute node at this file's colour attribute.
    name = color_attr_name(obj)
    print(f"[render]   colour attribute: '{name}' "
          f"(available: {[a.name for a in obj.data.color_attributes]})")
    for node in mat.node_tree.nodes:
        if node.type == "ATTRIBUTE":
            node.attribute_name = name
    mod = obj.modifiers.new("ToPoints", "NODES")
    mod.node_group = ng_factory
    bpy.context.scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(obj, do_unlink=True)


def main():
    args = parse_args()
    plys = sorted(glob.glob(os.path.join(args.ply, "*.ply")))
    if not plys:
        raise SystemExit(f"no .ply files in {args.ply}")
    os.makedirs(args.out, exist_ok=True)

    clear_scene()
    setup_world()
    setup_camera(args.cam_dist, args.cam_elev)
    setup_render(args)
    setup_compositor_glare(args.glare)

    mat = make_star_material(args.emission)
    ng = make_points_modifier_group(args.radius, mat)

    for i, ply in enumerate(plys):
        name = os.path.splitext(os.path.basename(ply))[0]
        out_png = os.path.abspath(os.path.join(args.out, name + ".png"))
        print(f"[render] {i + 1}/{len(plys)}  {name}")
        render_frame(ply, out_png, ng, mat)
    print(f"[render] done: {len(plys)} frames -> {args.out}/")


if __name__ == "__main__":
    main()
