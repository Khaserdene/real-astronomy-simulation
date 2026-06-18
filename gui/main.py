"""Astronomy Simulator -- PyQt6 desktop application.

    .venv/Scripts/python.exe -m gui.main

A window with a live 3D viewport and control panels: pick a scenario, set seed
and particle count, Build, then Run/Pause/Step.  Stars and gas are coloured by
physics (temperature, age, density).  The heavy lifting lives in the GUI-free
:class:`~core.sim_controller.SimController`; this module is just the view.
"""
from __future__ import annotations

import argparse
import os
import sys

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QComboBox, QSpinBox, QPushButton, QLabel, QSlider, QCheckBox, QFormLayout,
    QFileDialog,
)

from core.engine import init_taichi
from core.scenarios import SCENARIOS
from core.sim_controller import SimController

_N_PRESETS = {"Preview (5k)": 5000, "Low (10k)": 10000,
              "Medium (20k)": 20000, "High (40k)": 40000}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astronomy Simulator")
        self.resize(1500, 950)
        self.ctrl = SimController()
        self._taichi_ready = False

        from gui.viewport import Viewport
        self.viewport = Viewport()

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self._build_panel(), 0)
        layout.addWidget(self.viewport, 1)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

    # ------------------------------------------------------------------ panel
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(300)
        v = QVBoxLayout(panel)

        # --- scene ---
        scene_box = QGroupBox("Scene")
        sf = QFormLayout(scene_box)
        self.scenario_combo = QComboBox()
        for name, sc in SCENARIOS.items():
            self.scenario_combo.addItem(sc.label, name)
        self.scenario_combo.currentIndexChanged.connect(self._on_scenario)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(7)
        self.n_combo = QComboBox()
        for label in _N_PRESETS:
            self.n_combo.addItem(label)
        self.n_combo.setCurrentText("Medium (20k)")
        sf.addRow("Scenario", self.scenario_combo)
        sf.addRow("Seed", self.seed_spin)
        sf.addRow("Particles", self.n_combo)
        self.desc_label = QLabel(SCENARIOS["disk"].description)
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: gray; font-size: 11px;")
        sf.addRow(self.desc_label)
        v.addWidget(scene_box)

        # --- run controls ---
        run_box = QGroupBox("Run")
        rv = QVBoxLayout(run_box)
        self.build_btn = QPushButton("Build")
        self.build_btn.clicked.connect(self._on_build)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._on_run)
        self.step_btn = QPushButton("Step")
        self.step_btn.clicked.connect(self._on_step)
        rv.addWidget(self.build_btn)
        rv.addWidget(self.run_btn)
        rv.addWidget(self.step_btn)
        self.spf_slider, spf_row = _slider("Steps/frame", 1, 20, 2)
        self.spf_slider.valueChanged.connect(self._update_spf_label)
        rv.addLayout(spf_row)
        v.addWidget(run_box)

        # --- view ---
        view_box = QGroupBox("View")
        vv = QVBoxLayout(view_box)
        self.size_slider, size_row = _slider("Point size", 1, 8, 3)
        self.size_slider.valueChanged.connect(
            lambda val: self.viewport.set_point_size(val))
        vv.addLayout(size_row)
        v.addWidget(view_box)

        # --- output / checkpoints ---
        out_box = QGroupBox("Output")
        ov = QVBoxLayout(out_box)
        self.autosnap_check = QCheckBox("Auto-snapshot while running")
        self.autosnap_check.toggled.connect(
            lambda c: setattr(self.ctrl.s, "auto_snapshot", c))
        self.save_btn = QPushButton("Save snapshot")
        self.save_btn.clicked.connect(lambda: self._set_status(self.ctrl.save_snapshot()))
        self.load_btn = QPushButton("Load checkpoint...")
        self.load_btn.clicked.connect(self._on_load)
        ov.addWidget(self.autosnap_check)
        ov.addWidget(self.save_btn)
        ov.addWidget(self.load_btn)
        v.addWidget(out_box)

        # --- timeline (snapshot playback) ---
        tl_box = QGroupBox("Timeline (saved frames)")
        tv = QVBoxLayout(tl_box)
        self.refresh_btn = QPushButton("Refresh frames")
        self.refresh_btn.clicked.connect(self._refresh_frames)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.valueChanged.connect(self._on_timeline)
        self.frame_label = QLabel("no frames")
        self._frames = []
        tv.addWidget(self.refresh_btn)
        tv.addWidget(self.timeline)
        tv.addWidget(self.frame_label)
        v.addWidget(tl_box)

        # --- render ---
        rbox = QGroupBox("Render (Blender)")
        rl = QVBoxLayout(rbox)
        self.blender_btn = QPushButton("Open in Blender (interactive)")
        self.blender_btn.clicked.connect(self._on_open_blender)
        self.quick_btn = QPushButton("Quick render (headless)")
        self.quick_btn.clicked.connect(self._on_quick_render)
        rl.addWidget(self.blender_btn)
        rl.addWidget(self.quick_btn)
        v.addWidget(rbox)

        v.addStretch(1)
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        v.addWidget(self.stats_label)
        self.status_label = QLabel("press Build")
        self.status_label.setWordWrap(True)
        v.addWidget(self.status_label)
        return panel

    # --------------------------------------------------------------- handlers
    def _ensure_taichi(self):
        if not self._taichi_ready:
            init_taichi("cuda")
            self._taichi_ready = True

    def _on_scenario(self):
        name = self.scenario_combo.currentData()
        self.desc_label.setText(SCENARIOS[name].description)

    def _on_build(self):
        self._ensure_taichi()
        self.ctrl.s.scenario = self.scenario_combo.currentData()
        self.ctrl.s.seed = self.seed_spin.value()
        self.ctrl.s.n = _N_PRESETS[self.n_combo.currentText()]
        self._set_status(self.ctrl.build())
        self.run_btn.setText("Run")
        self._refresh_view()

    def _on_run(self):
        self.ctrl.toggle_run()
        self.run_btn.setText("Pause" if self.ctrl.s.running else "Run")

    def _on_step(self):
        self.ctrl.step_once(self.spf_slider.value())
        self._refresh_view()

    def _update_spf_label(self):
        pass  # label updates itself via _slider's connection

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load checkpoint", self.ctrl.s.out_dir, "Snapshots (*.h5)")
        if path:
            self._ensure_taichi()
            self._set_status(self.ctrl.load_checkpoint(path))
            self.run_btn.setText("Run")
            self._refresh_view()

    def _refresh_frames(self):
        self._frames = self.ctrl.list_snapshots()
        self.timeline.setRange(0, max(len(self._frames) - 1, 0))
        self.frame_label.setText(
            f"{len(self._frames)} frames" if self._frames else "no frames")

    def _on_timeline(self, idx):
        if self._frames and 0 <= idx < len(self._frames):
            self.ctrl.s.running = False
            self.run_btn.setText("Run")
            pos, rgba = SimController.display_snapshot(self._frames[idx])
            self.viewport.set_points(pos, rgba)
            self.frame_label.setText(os.path.basename(self._frames[idx]))

    def _on_open_blender(self):
        if self.ctrl.engine is None:
            self._set_status("build/run something first")
            return
        try:
            from export.to_blender import export_frame
            from gui.blender_launcher import open_in_blender
            snap = self.ctrl.save_snapshot()
            frame_dir = os.path.join(self.ctrl.s.out_dir, "blender_frame")
            export_frame(snap, frame_dir)
            open_in_blender(frame_dir)
            self._set_status("opening in Blender (interactive)...")
        except Exception as e:
            self._set_status(f"Blender error: {e}")

    def _on_quick_render(self):
        if self.ctrl.engine is None:
            self._set_status("build/run something first")
            return
        try:
            from export.to_pointcloud import snapshot_to_ply
            from gui.blender_launcher import quick_render
            snap = self.ctrl.save_snapshot()
            rdir = os.path.join(self.ctrl.s.out_dir, "_quick")
            ply_dir = os.path.join(rdir, "ply")
            os.makedirs(ply_dir, exist_ok=True)
            snapshot_to_ply(snap, os.path.join(ply_dir, "frame.ply"))
            self._set_status("rendering (blocks briefly)...")
            QApplication.processEvents()
            quick_render(ply_dir, os.path.join(rdir, "render"),
                         res=900, samples=32, **{"cam-elev": 72})
            png = os.path.join(rdir, "render", "frame.png")
            if os.path.exists(png):
                os.startfile(png)  # noqa: Windows
            self._set_status("quick render done")
        except Exception as e:
            self._set_status(f"render error: {e}")

    # ------------------------------------------------------------------ loop
    def _tick(self):
        if self.ctrl.s.running:
            self.ctrl.advance(self.spf_slider.value())
            self._refresh_view()
        self._refresh_stats()

    def _refresh_view(self):
        pos, rgba = self.ctrl.display_arrays()
        self.viewport.set_points(pos, rgba)

    def _refresh_stats(self):
        st = self.ctrl.stats()
        lines = [f"scenario : {st.get('scenario', '-')}",
                 f"step     : {st['step']}",
                 f"time     : {st['time']:.4f}",
                 f"N        : {st['n']:,}"]
        if st.get("energy") is not None:
            lines.append(f"energy   : {st['energy']:.4g}")
        if "stars" in st:
            lines.append(f"stars    : {st['stars']:,}")
        self.stats_label.setText("\n".join(lines))

    def _set_status(self, text: str):
        self.status_label.setText(text)


def _slider(label, lo, hi, val):
    """A labelled horizontal slider; returns (slider, row_layout)."""
    row = QHBoxLayout()
    name = QLabel(label)
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi); s.setValue(val)
    vallbl = QLabel(str(val))
    s.valueChanged.connect(lambda v: vallbl.setText(str(v)))
    row.addWidget(name); row.addWidget(s); row.addWidget(vallbl)
    return s, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="build+run N ticks, screenshot, then quit")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    if args.smoke:
        win._ensure_taichi()
        win.ctrl.s.scenario = "disk"
        win.ctrl.s.n = 10000
        win.ctrl.build()
        win.ctrl.s.running = True
        ticks = {"n": 0}

        def smoke_tick():
            ticks["n"] += 1
            if ticks["n"] >= args.smoke:
                os.makedirs("output", exist_ok=True)
                win.viewport.grabFramebuffer().save("output/gui_smoke.png")
                app.quit()
        t = QTimer(); t.timeout.connect(smoke_tick); t.start(40)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
