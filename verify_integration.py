import sys
import os
import time
import numpy as np

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt6.QtWidgets import QApplication

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f'  PASS  {name}')
        passed += 1
    except Exception as e:
        print(f'  FAIL  {name}: {e}')
        import traceback
        traceback.print_exc()
        failed += 1

def simple_mvp(dist=50.0, fov=60.0, w=800, h=600):
    import math
    aspect = w / h
    f = 1.0 / math.tan(math.radians(fov / 2))
    near, far = 0.1, 1000.0
    proj = np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
        [0, 0, -1, 0]
    ], dtype=np.float64)
    view = np.eye(4, dtype=np.float64)
    view[2, 3] = -dist
    return proj @ view

def test_brush_math():
    from gui.brush import world_to_screen, pick_radius, scatter_points, screen_ray_to_plane
    mvp = simple_mvp()
    w, h = 800, 600

    def t_world_to_screen():
        pts = np.array([[0.0, 0.0, 0.0]])
        sx, sy, vis = world_to_screen(pts, mvp, w, h)
        assert vis[0] == True
        assert abs(sx[0] - 400) < 1.0
        assert abs(sy[0] - 300) < 1.0
    test("world_to_screen center", t_world_to_screen)

    def t_pick_radius():
        pts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        mask = pick_radius(pts, mvp, w, h, 400, 300, 50.0)
        assert mask[0] == True
        assert mask[1] == False
    test("pick_radius", t_pick_radius)

    def t_scatter():
        pts = scatter_points(np.array([1.0, 2.0, 3.0]), 100, 5.0, seed=42)
        assert pts.shape == (100, 3)
        dist = np.linalg.norm(pts - np.array([1.0, 2.0, 3.0]), axis=1)
        assert np.all(dist <= 5.0)
    test("scatter_points", t_scatter)

    def t_ray():
        p = screen_ray_to_plane(400, 300, w, h, mvp, plane_z=0.0)
        assert p is not None
        assert abs(p[0]) < 1e-5 and abs(p[1]) < 1e-5 and abs(p[2]) < 1e-5
    test("screen_ray_to_plane", t_ray)

def test_scene_builder():
    from core.scene_spec import ObjectSpec, SceneSpec
    from core.objects import templates_for, compose_spec, build_scene, build_object
    from core.engine import init_taichi

    def t_templates():
        t = templates_for("galaxy")
        assert "disk_galaxy" in t
    test("templates_for galaxy", t_templates)

    def t_compose():
        objs = [ObjectSpec(template="disk_galaxy", n=1000, seed=1)]
        spec = compose_spec(objs, domain="galaxy")
        assert spec.engine_kind == "gravity"
        assert spec.primary.n == 1000
    test("compose_spec", t_compose)

    def t_build():
        init_taichi("cpu")
        spec = SceneSpec(engine_kind="gravity", dt=1e-4, softening=0.1, objects=[ObjectSpec(template="disk_galaxy", n=1000, seed=1)])
        engine, dt = build_scene(spec)
        assert engine.n == 1000
    test("build_scene", t_build)

def test_timeline():
    from gui.timeline import TimelineWidget
    
    app = QApplication.instance() or QApplication([])

    def t_set_frames():
        tw = TimelineWidget()
        tw.set_frames(["snap_0.h5", "snap_1.h5"])
        assert tw.slider.maximum() == 1
        assert tw.current_index() == 1
    test("Timeline.set_frames follow mode", t_set_frames)

    def t_scrub():
        tw = TimelineWidget()
        tw.set_frames(["snap_0.h5", "snap_1.h5", "snap_2.h5"])
        tw._scrub_to(0)
        assert tw.current_index() == 0
        assert not tw._follow
    test("Timeline scrub", t_scrub)

def test_full_pipeline():
    from core.engine import init_taichi
    from core.sim_controller import SimController
    from core.scene_spec import ObjectSpec, SceneSpec
    from gui.brush import scatter_points
    import tempfile

    init_taichi("cpu")

    def t_pipeline():
        ctrl = SimController()
        spec = SceneSpec(engine_kind="gravity", dt=1e-4, softening=0.1, objects=[ObjectSpec(template="disk_galaxy", n=1000, seed=1)])
        ctrl.build_scene(spec)
        
        ctrl.s.running = True
        ctrl.advance(5)
        pos, rgba = ctrl.display_arrays()
        assert ctrl.engine.n == 1000

        # edit add
        pts = scatter_points(np.array([0.,0.,0.]), 50, 2.0)
        ctrl.edit_particles(add_pos=pts, add_ptype=1)
        assert ctrl.engine.n == 1050

        # edit remove
        ctrl.edit_particles(remove_idx=np.array([0,1,2], dtype=np.int32))
        assert ctrl.engine.n == 1047

        # snapshot
        with tempfile.TemporaryDirectory() as td:
            ctrl.s.out_dir = td
            path = ctrl.save_snapshot()
            assert os.path.exists(path)
            
            snaps = ctrl.list_snapshots()
            assert len(snaps) == 1

            ctrl.step_once(1)
            ctrl.load_checkpoint(path)
            assert ctrl.engine.step_count == 5
    test("Full integration pipeline", t_pipeline)

def main():
    print("=== Group A: Brush math ===")
    test_brush_math()
    
    print("\n=== Group B: Scene builder ===")
    test_scene_builder()

    print("\n=== Group C: Timeline ===")
    test_timeline()

    print("\n=== Group D: Full pipeline ===")
    test_full_pipeline()

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

if __name__ == '__main__':
    main()
