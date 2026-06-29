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

import numpy as np

from PyQt6.QtCore import QTimer, Qt, QObject, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QComboBox, QSpinBox, QPushButton, QLabel, QSlider, QCheckBox, QFormLayout,
    QFileDialog, QMessageBox, QProgressBar,
)

from core.engine import init_taichi
from core.scenarios import SCENARIOS
from core.sim_controller import SimController

# Quick-pick particle counts (the spin box accepts any value in between).
_N_PRESETS = [("5k", 5000), ("10k", 10000), ("20k", 20000),
              ("40k", 40000), ("100k", 100000)]
_N_MIN, _N_MAX, _N_DEFAULT = 100, 2_000_000, 20000
_N_WARN = 50000  # above this, direct N^2 gravity gets slow

# Brush "add" particle types -> core.state ptype codes (DM=0, star=1, gas=2).
_PTYPE = {"Star": 1, "Gas": 2, "Dark matter": 0}
_CLIP_KPC = 120.0  # matches sim_display: only pick what is actually drawn


class Sparkline(QWidget):
    """Tiny self-painted trend plot for a scalar diagnostic over time.

    Keeps a rolling history and draws it as a line, with an optional reference
    level (e.g. virial Q = 1) marked.  No external plotting dependency.
    """

    def __init__(self, label="", ref=None, lo=0.0, hi=2.0, maxlen=240, parent=None):
        super().__init__(parent)
        self.label, self.ref, self.lo, self.hi = label, ref, lo, hi
        self._hist: list[float] = []
        self._maxlen = maxlen
        self.setMinimumHeight(46)

    def clear(self):
        self._hist.clear(); self.update()

    def push(self, value):
        if value is None:
            return
        self._hist.append(float(value))
        if len(self._hist) > self._maxlen:
            self._hist = self._hist[-self._maxlen:]
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(20, 22, 28))
        lo, hi = self.lo, self.hi
        if self._hist:
            lo = min(lo, min(self._hist)); hi = max(hi, max(self._hist))
        span = (hi - lo) or 1.0

        def y_of(v):
            return h - 4 - (v - lo) / span * (h - 8)

        if self.ref is not None and lo <= self.ref <= hi:
            p.setPen(QPen(QColor(90, 90, 110), 1, Qt.PenStyle.DashLine))
            yr = int(y_of(self.ref))
            p.drawLine(0, yr, w, yr)
        if len(self._hist) >= 2:
            p.setPen(QPen(QColor(90, 200, 255), 1))
            n = len(self._hist)
            for i in range(1, n):
                x0 = int((i - 1) / (n - 1) * (w - 1))
                x1 = int(i / (n - 1) * (w - 1))
                p.drawLine(x0, int(y_of(self._hist[i - 1])),
                           x1, int(y_of(self._hist[i])))
        p.setPen(QColor(180, 185, 195))
        txt = self.label
        if self._hist:
            txt += "  %.2f" % self._hist[-1]
        p.drawText(4, 13, txt)
        p.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astronomy Simulator")
        self.resize(1500, 950)
        self.ctrl = SimController()
        self._taichi_ready = False

        from gui.viewport import Viewport
        from gui.timeline import TimelineWidget
        self.viewport = Viewport()
        self.viewport.eraseStroke.connect(self._on_erase_stroke)
        self.viewport.addAt.connect(self._on_add_at)

        # The control panel is taller than a 1080p screen, so put it in a scroll
        # area -- it scrolls instead of squashing the buttons when space is tight.
        from PyQt6.QtWidgets import QScrollArea
        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setWidget(self._build_panel())
        panel_scroll.setFixedWidth(322)
        panel_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        top = QWidget()
        top_row = QHBoxLayout(top)
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.addWidget(panel_scroll, 0)
        top_row.addWidget(self.viewport, 1)

        self.timeline = TimelineWidget()
        self.timeline.frameSelected.connect(self._show_saved_frame)
        self.timeline.goLive.connect(self._go_live)
        self.timeline.refreshRequested.connect(self._refresh_timeline)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(top, 1)
        layout.addWidget(self.timeline, 0)
        self.setCentralWidget(central)

        self._snap_seen = 0  # last snapshot count pushed to the timeline
        self._busy = False   # a blocking task (render) is running off-thread
        self._task = None    # keep a ref so the async task isn't GC'd

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
        # Curated presets: one click loads a tuned, good-looking configuration.
        from core.presets import PRESETS
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("— custom —", None)
        for _pname in PRESETS:
            self.preset_combo.addItem(_pname, _pname)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        sf.addRow("Preset", self.preset_combo)
        self.scenario_combo = QComboBox()
        for name, sc in SCENARIOS.items():
            self.scenario_combo.addItem(sc.label, name)
        self.scenario_combo.currentIndexChanged.connect(self._on_scenario)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(7)
        sf.addRow("Scenario", self.scenario_combo)
        sf.addRow("Seed", self.seed_spin)
        
        self.live_halo_check = QCheckBox("Live Halo (N-body Dark Matter)")
        self.live_halo_check.setToolTip("Use N-body particles for Dark Matter instead of analytic potential (where supported).")
        self.live_halo_check.setChecked(True)
        self.live_halo_check.toggled.connect(
            lambda c: setattr(self.ctrl.s, "live_halo", c))
        sf.addRow(self.live_halo_check)

        # --- particle count: free spin box + quick presets ---
        self.n_spin = QSpinBox()
        self.n_spin.setRange(_N_MIN, _N_MAX)
        self.n_spin.setGroupSeparatorShown(True)
        self.n_spin.setSingleStep(1000)
        self.n_spin.setValue(_N_DEFAULT)
        self.n_spin.valueChanged.connect(self._update_n_warn)
        sf.addRow("Particles", self.n_spin)
        preset_row = QHBoxLayout()
        for label, value in _N_PRESETS:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda _=False, val=value: self.n_spin.setValue(val))
            preset_row.addWidget(btn)
        sf.addRow(preset_row)
        self.n_warn = QLabel("")
        self.n_warn.setWordWrap(True)
        self.n_warn.setStyleSheet("color: #c47f00; font-size: 10px;")
        sf.addRow(self.n_warn)

        # --- scenario description + details ---
        self.desc_label = QLabel(SCENARIOS["disk"].description)
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: gray; font-size: 11px;")
        sf.addRow(self.desc_label)
        self.details_btn = QPushButton("Details …")
        self.details_btn.clicked.connect(self._show_details)
        sf.addRow(self.details_btn)

        self.builder_btn = QPushButton("Scene builder… (compose objects)")
        self.builder_btn.clicked.connect(self._on_scene_builder)
        sf.addRow(self.builder_btn)
        self.physics_btn = QPushButton("Physics settings…")
        self.physics_btn.clicked.connect(self._on_physics)
        sf.addRow(self.physics_btn)

        self._update_n_warn(self.n_spin.value())
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
        self.bh_check = QCheckBox("Fast gravity (Barnes-Hut) — applies on Build")
        self.bh_check.setToolTip("Treecode O(N log N) gravity for large N "
                                 "(gravity scenarios). Slightly approximate.")
        self.bh_check.toggled.connect(
            lambda c: setattr(self.ctrl.s, "gravity_mode", "bh" if c else "direct"))
        rv.addWidget(self.bh_check)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("frame %p%")
        rv.addWidget(self.progress)
        v.addWidget(run_box)

        # --- view ---
        view_box = QGroupBox("View")
        vv = QVBoxLayout(view_box)
        self.size_slider, size_row = _slider("Point size", 1, 8, 3)
        self.size_slider.valueChanged.connect(
            lambda val: self.viewport.set_point_size(val))
        vv.addLayout(size_row)
        v.addWidget(view_box)

        # --- brush (interactive particle editing) ---
        brush_box = QGroupBox("Brush (edit particles)")
        bf = QFormLayout(brush_box)
        self.brush_check = QCheckBox("Enable (left-drag in viewport)")
        self.brush_check.toggled.connect(self._on_brush_toggle)
        bf.addRow(self.brush_check)
        self.brush_mode_combo = QComboBox()
        self.brush_mode_combo.addItems(["Erase", "Add"])
        self.brush_mode_combo.currentTextChanged.connect(self._on_brush_mode)
        bf.addRow("Mode", self.brush_mode_combo)
        self.brush_radius_slider, br_row = _slider("Radius px", 5, 80, 25)
        self.brush_radius_slider.valueChanged.connect(
            lambda val: setattr(self.viewport, "brush_radius_px", float(val)))
        bf.addRow(br_row)
        self.add_type_combo = QComboBox()
        self.add_type_combo.addItems(list(_PTYPE))
        bf.addRow("Add type", self.add_type_combo)
        self.add_count_spin = QSpinBox()
        self.add_count_spin.setRange(10, 20000)
        self.add_count_spin.setSingleStep(100); self.add_count_spin.setValue(500)
        bf.addRow("Add count", self.add_count_spin)
        from PyQt6.QtWidgets import QDoubleSpinBox
        self.add_radius_spin = QDoubleSpinBox()
        self.add_radius_spin.setRange(0.2, 30.0); self.add_radius_spin.setValue(3.0)
        bf.addRow("Add radius kpc", self.add_radius_spin)
        # Work plane: Add drops on z = plane; Erase removes near that layer.
        self.plane_slider, plane_row = _slider("Plane Z kpc", -60, 60, 0)
        self.plane_slider.valueChanged.connect(
            lambda val: setattr(self.viewport, "brush_plane_z", float(val)))
        bf.addRow(plane_row)
        v.addWidget(brush_box)

        # --- output / checkpoints ---
        out_box = QGroupBox("Output")
        ov = QVBoxLayout(out_box)
        self.folder_label = QLabel("")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("font-size: 10px; color: gray;")
        self._update_folder_label()
        self.folder_btn = QPushButton("Set output folder…")
        self.folder_btn.clicked.connect(self._on_set_folder)
        self.open_folder_btn = QPushButton("Open simulation folder…")
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        self.autosnap_check = QCheckBox("Auto-snapshot while running")
        self.autosnap_check.toggled.connect(
            lambda c: setattr(self.ctrl.s, "auto_snapshot", c))
        # How many simulation steps each saved frame represents. Smaller = more
        # frames (smoother playback, finer time sampling, more disk); larger =
        # fewer frames covering more evolution per frame.
        self.snap_every_spin = QSpinBox()
        self.snap_every_spin.setRange(1, 100000)
        self.snap_every_spin.setValue(self.ctrl.s.snap_every)
        self.snap_every_spin.setGroupSeparatorShown(True)
        self.snap_every_spin.valueChanged.connect(
            lambda v: setattr(self.ctrl.s, "snap_every", v))
        snap_row = QHBoxLayout()
        snap_row.addWidget(QLabel("Steps / saved frame"))
        snap_row.addWidget(self.snap_every_spin)
        self.save_btn = QPushButton("Save snapshot")
        self.save_btn.clicked.connect(self._on_save)
        self.load_btn = QPushButton("Load checkpoint...")
        self.load_btn.clicked.connect(self._on_load)
        ov.addWidget(self.folder_label)
        ov.addWidget(self.folder_btn)
        ov.addWidget(self.open_folder_btn)
        ov.addWidget(self.autosnap_check)
        ov.addLayout(snap_row)
        ov.addWidget(self.save_btn)
        ov.addWidget(self.load_btn)
        v.addWidget(out_box)

        # (Snapshot playback lives in the bottom timeline bar, not here.)

        # --- render ---
        rbox = QGroupBox("Render (Blender)")
        rl = QVBoxLayout(rbox)
        self.blender_btn = QPushButton("Open in Blender (current frame)")
        self.blender_btn.clicked.connect(self._on_open_blender)
        self.anim_btn = QPushButton("Open in Blender (animation)")
        self.anim_btn.clicked.connect(self._on_open_blender_animation)
        self.quick_btn = QPushButton("Quick render (headless)")
        self.quick_btn.clicked.connect(self._on_quick_render)
        # Dark matter dominates the particle count and can bury the visible
        # galaxy; let the user hide it in the live viewport.
        self.show_dm_check = QCheckBox("Show dark matter")
        self.show_dm_check.setChecked(True)
        self.show_dm_check.toggled.connect(lambda _c: self._refresh_view())
        rl.addWidget(self.blender_btn)
        rl.addWidget(self.anim_btn)
        rl.addWidget(self.quick_btn)
        rl.addWidget(self.show_dm_check)
        v.addWidget(rbox)

        v.addStretch(1)
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        v.addWidget(self.stats_label)
        # Live virial-Q trend (Q=1 = equilibrium reference line).
        self.virial_plot = Sparkline("virial Q", ref=1.0, lo=0.0, hi=2.0)
        v.addWidget(self.virial_plot)
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
        self._update_n_warn(self.n_spin.value())

    def _update_n_warn(self, n: int):
        name = self.scenario_combo.currentData() or "disk"
        kind = SCENARIOS[name].kind
        if n >= _N_WARN:
            cost = "direct N² gravity" if kind == "gravity" else "SPH neighbour search"
            self.n_warn.setText(
                f"⚠ {n:,} particles — {cost} may be slow at this count.")
        else:
            self.n_warn.setText("")

    def _show_details(self):
        name = self.scenario_combo.currentData()
        sc = SCENARIOS[name]
        QMessageBox.information(self, sc.label,
                               sc.details or sc.description or "(no details)")

    def _on_scene_builder(self):
        from gui.scene_builder import SceneBuilderDialog
        from PyQt6.QtWidgets import QDialog
        dlg = SceneBuilderDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            spec = dlg.result_spec()
        except ValueError as e:
            self._set_status(f"scene error: {e}")
            return
        self._ensure_taichi()
        self._set_status(self.ctrl.build_scene(spec))
        self.timeline.follow_live()
        self._refresh_timeline()
        self._refresh_view()

    def _on_preset(self):
        name = self.preset_combo.currentData()
        if not name:
            return
        from core.presets import apply_preset
        apply_preset(self.ctrl.s, name)
        i = self.scenario_combo.findData(self.ctrl.s.scenario)
        if i >= 0:
            self.scenario_combo.setCurrentIndex(i)
        self.n_spin.setValue(self.ctrl.s.n)
        self.seed_spin.setValue(self.ctrl.s.seed)
        self._set_status(f"preset '{name}' loaded — press Build")

    def _on_physics(self):
        from gui.physics_dialog import PhysicsDialog
        from PyQt6.QtWidgets import QDialog
        dlg = PhysicsDialog(self.ctrl.s, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._set_status("physics settings updated — press Build")

    def _on_build(self):
        # Guard: rebuilding after continuing a loaded folder with a different N
        # cannot extend the existing frames -- offer to overwrite from scratch.
        if self.ctrl.continuing and self.n_spin.value() != self.ctrl.s.n:
            reply = QMessageBox.question(
                self, "Particle count changed",
                f"This folder's frames have N={self.ctrl.s.n:,}, but you set "
                f"N={self.n_spin.value():,}.\n\nA new count cannot continue the "
                "existing frames. Restart from scratch and overwrite this "
                "folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                self._set_status("build cancelled (N unchanged to continue)")
                return
            self._wipe_snapshots(self.ctrl.s.out_dir)
        self._ensure_taichi()
        self.ctrl.s.scenario = self.scenario_combo.currentData()
        self.ctrl.s.seed = self.seed_spin.value()
        self.ctrl.s.n = self.n_spin.value()
        self._set_status(self.ctrl.build())
        self.run_btn.setText("Run")
        self.virial_plot.clear()
        self.timeline.follow_live()
        self._refresh_timeline()
        self._refresh_view()

    def _on_set_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.ctrl.s.out_dir)
        if folder:
            self.ctrl.s.out_dir = folder
            self.ctrl.continuing = False
            self._update_folder_label()
            self._set_status(f"output -> {folder}")

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Open simulation folder", self.ctrl.s.out_dir)
        if not folder:
            return
        self._ensure_taichi()
        self._set_status(self.ctrl.open_folder(folder))
        # Reflect the resumed scene back into the controls.
        self._sync_controls_from_ctrl()
        self.run_btn.setText("Run")
        self._update_folder_label()
        self._refresh_timeline()
        self._refresh_view()

    def _sync_controls_from_ctrl(self):
        idx = self.scenario_combo.findData(self.ctrl.s.scenario)
        if idx >= 0:
            self.scenario_combo.setCurrentIndex(idx)
        self.seed_spin.setValue(self.ctrl.s.seed)
        self.n_spin.setValue(self.ctrl.s.n)

    def _update_folder_label(self):
        self.folder_label.setText(f"folder: {self.ctrl.s.out_dir}")

    def _wipe_snapshots(self, folder):
        for p in self.ctrl.list_snapshots(folder):
            try:
                os.remove(p)
            except OSError:
                pass

    def _on_run(self):
        self.ctrl.toggle_run()
        self.run_btn.setText("Pause" if self.ctrl.s.running else "Run")
        if self.ctrl.s.running:
            self.timeline.stop()            # leave playback
            self.timeline.follow_live()     # ride the newest saved frame again

    def _on_step(self):
        self.ctrl.step_once(self.spf_slider.value())
        self._refresh_view()

    def _update_spf_label(self):
        pass  # label updates itself via _slider's connection

    # -------------------------------------------------------------------- brush
    def _on_brush_toggle(self, on: bool):
        self.viewport.brush_enabled = on
        if on and self.ctrl.s.running:           # freeze the scene while editing
            self.ctrl.s.running = False
            self.run_btn.setText("Run")
        mode = self.brush_mode_combo.currentText().lower()
        self._set_status(f"brush ON — left-drag to {mode}" if on else "brush off")

    def _on_brush_mode(self, text: str):
        self.viewport.brush_mode = "add" if text == "Add" else "erase"

    def _on_erase_stroke(self, samples, mvp, w, h, radius):
        if self.ctrl.engine is None:
            return
        from gui.brush import pick_radius
        pos = self.ctrl.engine.to_state().pos
        within = np.linalg.norm(pos, axis=1) < _CLIP_KPC
        # Restrict erase to the work-plane layer (a slab around z = plane) so it
        # stays consistent with the Add plane and doesn't punch through depth.
        plane_z = self.viewport.brush_plane_z
        slab = max(self.add_radius_spin.value() * 2.0, 6.0)
        within &= np.abs(pos[:, 2] - plane_z) < slab
        mask = np.zeros(len(pos), bool)
        for (mx, my) in samples:
            mask |= pick_radius(pos, mvp, w, h, mx, my, radius)
        mask &= within
        if mask.any():
            self._set_status(self.ctrl.edit_particles(remove_idx=np.where(mask)[0]))
            self._refresh_view()

    def _on_add_at(self, center):
        from gui.brush import scatter_points
        pts = scatter_points(np.asarray(center, float),
                             self.add_count_spin.value(),
                             self.add_radius_spin.value())
        ptype = _PTYPE[self.add_type_combo.currentText()]
        self._set_status(self.ctrl.edit_particles(add_pos=pts, add_ptype=ptype))
        self._refresh_view()

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load checkpoint", self.ctrl.s.out_dir, "Snapshots (*.h5)")
        if path:
            self._ensure_taichi()
            self._set_status(self.ctrl.load_checkpoint(path))
            self._sync_controls_from_ctrl()
            self.run_btn.setText("Run")
            self._update_folder_label()
            self._refresh_timeline()
            self._refresh_view()

    def _on_save(self):
        self._set_status(self.ctrl.save_snapshot())
        self._refresh_timeline()

    def _refresh_timeline(self):
        frames = self.ctrl.list_snapshots()
        self._snap_seen = len(frames)
        self.timeline.set_frames(frames)

    def _show_saved_frame(self, idx):
        """User scrubbed/played to a saved frame -- pause live, show it."""
        frames = self.ctrl.list_snapshots()
        if not (0 <= idx < len(frames)):
            return
        if self.ctrl.s.running:
            self.ctrl.s.running = False
            self.run_btn.setText("Run")
        pos, rgba = SimController.display_snapshot(frames[idx])
        self.viewport.set_points(pos, rgba)

    def _go_live(self):
        """Playback reached the end (or 'Live' pressed) -- continue stepping."""
        self.timeline.follow_live()
        if self.ctrl.engine is None:
            frames = self.ctrl.list_snapshots()
            if frames:
                self._ensure_taichi()
                self._set_status(self.ctrl.open_folder(self.ctrl.s.out_dir))
                self._sync_controls_from_ctrl()
            else:
                return
        self.ctrl.s.running = True
        self.run_btn.setText("Pause")
        self._refresh_view()

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

    def _on_open_blender_animation(self):
        """Hand the whole run's snapshot folder to Blender as an animation."""
        frames = self.ctrl.list_snapshots()
        if not frames:
            self._set_status("no saved frames -- enable auto-snapshot and run, "
                             "or Save snapshot first")
            return
        try:
            from gui.blender_launcher import open_animation_in_blender
            open_animation_in_blender(self.ctrl.s.out_dir)
            self._set_status(f"opening {len(frames)} frames in Blender…")
        except Exception as e:
            self._set_status(f"Blender error: {e}")

    def _on_quick_render(self):
        if self.ctrl.engine is None:
            self._set_status("build/run something first")
            return
        if self._busy:
            self._set_status("already rendering…")
            return
        try:
            from export.to_pointcloud import snapshot_to_ply
            from gui.blender_launcher import quick_render
            # PLY prep is fast -> do it on the UI thread, then render off-thread.
            snap = self.ctrl.save_snapshot()
            rdir = os.path.join(self.ctrl.s.out_dir, "_quick")
            ply_dir = os.path.join(rdir, "ply")
            os.makedirs(ply_dir, exist_ok=True)
            snapshot_to_ply(snap, os.path.join(ply_dir, "frame.ply"))
        except Exception as e:
            self._set_status(f"render prep error: {e}")
            return

        def work():
            quick_render(ply_dir, os.path.join(rdir, "render"),
                         res=900, samples=32, **{"cam-elev": 72})
            return os.path.join(rdir, "render", "frame.png")

        self._set_busy(True, "rendering (Blender, off-thread)…")
        self._task = _AsyncTask(work)
        self._task.finished.connect(self._on_render_done)
        self._task.start()

    def _on_render_done(self, result):
        self._set_busy(False)
        self._task = None
        if isinstance(result, Exception):
            self._set_status(f"render error: {result}")
            return
        if result and os.path.exists(result):
            os.startfile(result)  # noqa: Windows
        self._set_status("quick render done")

    def _set_busy(self, on: bool, text: str = ""):
        self._busy = on
        for b in (self.quick_btn, self.blender_btn, self.anim_btn):
            b.setEnabled(not on)
        if on:
            self.progress.setRange(0, 0)        # indeterminate "busy" sweep
            self.progress.setFormat("%p%")
            if text:
                self._set_status(text)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("frame %p%")

    # ------------------------------------------------------------------ loop
    def _tick(self):
        if self.ctrl.s.running:
            self.ctrl.advance(self.spf_slider.value())
            self._refresh_view()
            if not self._busy:
                self.progress.setValue(int(self.ctrl.frame_progress() * 100))
            # Auto-snapshotting drops new frames; let the timeline grow with them
            # (cheap glob, only when the count actually changed).
            if self.ctrl.s.auto_snapshot:
                if len(self.ctrl.list_snapshots()) != self._snap_seen:
                    self._refresh_timeline()
        self._refresh_stats()

    def _refresh_view(self):
        show_dm = self.show_dm_check.isChecked()
        pos, rgba = self.ctrl.display_arrays(show_dm=show_dm)
        self.viewport.set_points(pos, rgba)

    def _refresh_stats(self):
        st = self.ctrl.stats()
        lines = [f"scenario : {st.get('scenario', '-')}",
                 f"step     : {st['step']}",
                 f"time     : {st['time']:.4f}",
                 f"N        : {st['n']:,}"]
        if st.get("energy") is not None:
            lines.append(f"energy   : {st['energy']:.4g}")
        if st.get("virial") is not None:
            # 2(KE+U_th)/|PE|: <1 collapsing, ~1 in equilibrium, >1 unbound.
            lines.append(f"virial Q : {st['virial']:.3f}")
        if st.get("thermal") is not None and st["thermal"] > 0.0:
            lines.append(f"thermal  : {st['thermal']:.4g}")
        if st.get("substeps") is not None:
            lines.append(f"substeps : {st['substeps']}")
        if st.get("toomre_q") is not None:
            # <1 unstable, 1-2 spiral/bar-forming, >>2 featureless.
            lines.append(f"Toomre Q : {st['toomre_q']:.2f}")
        if st.get("v_rot") is not None:
            lines.append(f"v_rot(8) : {st['v_rot']:.0f} km/s")
        if st.get("smbh_mass") is not None:
            lines.append(f"SMBH M   : {st['smbh_mass']:.4g}")
        # --- Gas / Stars block ---
        if st.get("gas_count") is not None:
            lines.append("--- gas / stars ---")
            lines.append(f"gas      : {st['gas_count']:,}  (M={st.get('gas_mass', 0):.3g})")
            lines.append(f"stars    : {st.get('star_count', st.get('stars', 0)):,}"
                         f"  (M={st.get('star_mass', 0):.3g})")
            if st.get("gas_fraction") is not None:
                lines.append(f"gas frac : {st['gas_fraction']:.3f}")
            if st.get("sfr") is not None:
                lines.append(f"SFR      : {st['sfr']:.3g}")
            if st.get("gas_metal") is not None:
                lines.append(f"Z(gas)   : {st['gas_metal']:.4f}")
        elif "stars" in st:
            lines.append(f"stars    : {st['stars']:,}")
        # --- Event counters ---
        if st.get("n_sn") is not None:
            ev = (f"SNe {st.get('n_sn', 0):,}  hyper {st.get('n_hypernova', 0):,}"
                  f"  BH {st.get('n_bh_formed', 0):,}")
            if st.get("n_ejected"):
                ev += f"  ejected {st['n_ejected']:,}"
            lines.append("--- events ---")
            lines.append(ev)
        self.stats_label.setText("\n".join(lines))
        if self.ctrl.s.running:
            self.virial_plot.push(st.get("virial"))

    def _set_status(self, text: str):
        self.status_label.setText(text)


class _AsyncTask(QObject):
    """Run a blocking callable off the UI thread; deliver result via a signal.

    ``finished`` carries the callable's return value, or the ``Exception`` it
    raised, marshalled back onto the main thread by Qt's queued connection.
    """
    finished = pyqtSignal(object)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def start(self):
        import threading

        def run():
            try:
                result = self._fn()
            except Exception as e:        # report back, don't crash the thread
                result = e
            self.finished.emit(result)

        threading.Thread(target=run, daemon=True).start()


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
