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

        # --- output ---
        out_box = QGroupBox("Output")
        ov = QVBoxLayout(out_box)
        self.autosnap_check = QCheckBox("Auto-snapshot while running")
        self.autosnap_check.toggled.connect(
            lambda c: setattr(self.ctrl.s, "auto_snapshot", c))
        self.save_btn = QPushButton("Save snapshot")
        self.save_btn.clicked.connect(lambda: self._set_status(self.ctrl.save_snapshot()))
        ov.addWidget(self.autosnap_check)
        ov.addWidget(self.save_btn)
        v.addWidget(out_box)

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
