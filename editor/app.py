"""Standalone galaxy-simulator editor (Taichi GGUI).

A live 3D viewport over the headless :class:`EditorController`: build a galaxy,
run/pause it, orbit the camera, checkpoint and resume, promote a preview to a
production run, and export frames for Blender -- all from one window.

Run:
    .venv/Scripts/python.exe -m editor.app

Smoke-test (open, run a few frames, auto-close -- needs a display):
    .venv/Scripts/python.exe -m editor.app --smoke 30
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import taichi as ti

from editor.controller import EditorController, LEVELS


class GalaxyView:
    """Owns the (re-allocatable) GPU fields the GGUI scene renders."""

    def __init__(self):
        self.tree = None
        self.pos = self.col = None
        self.n = 0

    def allocate(self, n: int) -> None:
        if n == self.n:
            return
        if self.tree is not None:
            self.tree.destroy()
        fb = ti.FieldsBuilder()
        self.pos = ti.Vector.field(3, ti.f32)
        self.col = ti.Vector.field(3, ti.f32)
        fb.dense(ti.i, max(n, 1)).place(self.pos, self.col)
        self.tree = fb.finalize()
        self.n = n

    def upload(self, pos: np.ndarray, col: np.ndarray) -> None:
        self.allocate(pos.shape[0])
        if pos.shape[0]:
            self.pos.from_numpy(pos)
            self.col.from_numpy(col)


class EditorApp:
    def __init__(self, steps_per_frame: int = 2):
        self.ctrl = EditorController()
        self.view = GalaxyView()
        self.steps_per_frame = steps_per_frame
        self._fps_t = time.time()
        self._fps = 0.0

    # ------------------------------------------------------------------ panel
    def _draw_panel(self, window):
        gui = window.get_gui()
        s = self.ctrl.s
        with gui.sub_window("Astronomy Simulator", 0.0, 0.0, 0.27, 1.0) as w:
            w.text("--- scene ---")
            if w.button("IC: " + s.ic):
                order = ["disk", "merger", "plummer"]
                s.ic = order[(order.index(s.ic) + 1) % len(order)]
            s.seed = int(w.slider_int("seed", int(s.seed), 0, 99))
            li = w.slider_int("level", LEVELS.index(s.level), 0, len(LEVELS) - 1)
            s.level = LEVELS[int(li)]
            if w.button("Build"):
                self.ctrl.build()
            if w.button("Promote -> high (same seed)"):
                self.ctrl.promote("high")

            w.text("--- run ---")
            if w.button("Pause" if s.running else "Run"):
                self.ctrl.toggle_run()
            if w.button("Step x10"):
                self.ctrl.step_once(10)
            self.steps_per_frame = int(
                w.slider_int("steps/frame", self.steps_per_frame, 1, 20))

            w.text("--- output ---")
            s.auto_snapshot = w.checkbox("auto-snapshot", s.auto_snapshot)
            s.snap_every = int(w.slider_int("snap every", s.snap_every, 5, 200))
            if w.button("Save snapshot"):
                self.ctrl.save_snapshot()
            if w.button("Load latest checkpoint"):
                self._load_latest()
            if w.button("Export PLYs (for Blender)"):
                self.ctrl.export_plys()

            w.text("--- view ---")
            s.point_scale = w.slider_float("zoom", s.point_scale, 0.1, 3.0)

            w.text("--- stats ---")
            st = self.ctrl.stats()
            w.text(f"step {st['step']}  t={st['time']:.4f}")
            w.text(f"N={st['n']:,}  {st.get('solver', '')}")
            if st["energy"] is not None:
                w.text(f"E={st['energy']:.4g}")
            w.text(f"fps {self._fps:.1f}")
            w.text(self.ctrl.status)

    def _load_latest(self):
        snaps = sorted(glob.glob(os.path.join(self.ctrl.s.out_dir, "snap_*.h5")))
        if snaps:
            self.ctrl.load_checkpoint(snaps[-1])

    # ------------------------------------------------------------------- loop
    def run(self, smoke_frames: int = 0):
        window = ti.ui.Window("Astronomy Simulator", (1400, 900), vsync=True)
        canvas = window.get_canvas()
        canvas.set_background_color((0.0, 0.0, 0.0))
        scene = window.get_scene() if hasattr(window, "get_scene") else ti.ui.Scene()
        camera = ti.ui.Camera()
        camera.position(0.0, -55.0, 35.0)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 0.0, 1.0)

        if smoke_frames:                     # auto-build a small run for testing
            self.ctrl.s.level = "low"
            self.ctrl.build()
            self.ctrl.s.running = True

        frames = 0
        while window.running:
            self.ctrl.advance(self.steps_per_frame)
            pos, col = self.ctrl.display_arrays()
            self.view.upload(pos, col)
            if smoke_frames and frames == 0 and pos.shape[0]:
                print(f"[smoke] N={pos.shape[0]} pos range "
                      f"[{pos.min():.1f},{pos.max():.1f}] colmax={col.max():.2f}")

            if smoke_frames:                       # deterministic framing
                camera.position(0.0, -55.0, 35.0)
                camera.lookat(0.0, 0.0, 0.0)
                camera.up(0.0, 0.0, 1.0)
            else:
                camera.track_user_inputs(window, movement_speed=0.5,
                                         hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((1.0, 1.0, 1.0))   # show emission colours fully
            scene.point_light(pos=(0.0, 0.0, 60.0), color=(1.0, 1.0, 1.0))
            if self.view.n:
                scene.particles(self.view.pos,
                                radius=0.4 * self.ctrl.s.point_scale,
                                per_vertex_color=self.view.col)
            canvas.scene(scene)
            self._draw_panel(window)
            window.show()

            self._tick_fps()
            frames += 1
            if smoke_frames and frames >= smoke_frames:
                os.makedirs("output", exist_ok=True)
                try:
                    buf = window.get_image_buffer_as_numpy()
                    import matplotlib.image as mpimg
                    mpimg.imsave("output/editor_smoke.png",
                                 np.flipud(np.transpose(buf, (1, 0, 2))))
                except Exception as e:
                    print(f"[smoke] buffer capture failed: {e}")
                    window.save_image("output/editor_smoke.png")
                break
        window.destroy()

    def _tick_fps(self):
        now = time.time()
        dt = now - self._fps_t
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._fps_t = now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cuda", choices=["cuda", "vulkan"])
    ap.add_argument("--smoke", type=int, default=0,
                    help="auto-run N frames then close (needs a display)")
    args = ap.parse_args()
    ti.init(arch=ti.cuda if args.arch == "cuda" else ti.vulkan, default_fp=ti.f64)
    EditorApp().run(smoke_frames=args.smoke)


if __name__ == "__main__":
    main()
