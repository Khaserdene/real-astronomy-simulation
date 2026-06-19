"""Physics settings dialog -- the engine-level (not structural) parameters.

Separate from the per-object structural parameters (masses, scales -- those live
in the scene builder), this panel edits how the *engine* behaves: the gravity
solver, radiative cooling of gas, star formation and supernova feedback.  Changes
apply to the next Build.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QGroupBox, QComboBox, QDoubleSpinBox,
    QCheckBox, QDialogButtonBox, QLabel,
)


class PhysicsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Physics settings")
        self.s = settings
        v = QVBoxLayout(self)

        info = QLabel("Engine-level physics. Applies to the next Build. "
                      "(Structural params like masses/scales are in the scene "
                      "builder.)")
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(info)

        # --- gravity ---
        g = QGroupBox("Gravity")
        gf = QFormLayout(g)
        self.grav_mode = QComboBox()
        self.grav_mode.addItem("Direct N² (exact)", "direct")
        self.grav_mode.addItem("Barnes-Hut tree (fast, large N)", "bh")
        self.grav_mode.setCurrentIndex(0 if self.s.gravity_mode == "direct" else 1)
        self.theta = self._spin(0.1, 1.5, 0.05, 3, self.s.theta)
        gf.addRow("Solver", self.grav_mode)
        gf.addRow("BH opening angle θ", self.theta)
        v.addWidget(g)

        # --- gas cooling ---
        c = QGroupBox("Gas radiative cooling")
        cf = QFormLayout(c)
        self.cooling = QCheckBox("Enabled")
        self.cooling.setChecked(bool(self.s.cooling))
        self.u_floor = self._spin(1.0, 5000.0, 5.0, 1, self.s.u_floor)
        self.t_cool = self._spin(0.001, 1.0, 0.005, 4, self.s.t_cool)
        cf.addRow(self.cooling)
        cf.addRow("Temperature floor u_floor", self.u_floor)
        cf.addRow("Cooling time t_cool", self.t_cool)
        v.addWidget(c)

        # --- star formation + feedback ---
        s = QGroupBox("Star formation & feedback (living/full galaxy)")
        sf = QFormLayout(s)
        self.sf_prob = self._spin(0.0, 1.0, 0.005, 3, self.s.sf_prob)
        self.du_sn = self._spin(0.0, 5000.0, 50.0, 1, self.s.du_sn)
        sf.addRow("SF probability / step", self.sf_prob)
        sf.addRow("Supernova energy du_sn", self.du_sn)
        v.addWidget(s)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _spin(self, lo, hi, step, decimals, val):
        s = QDoubleSpinBox()
        s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(decimals)
        s.setValue(float(val))
        return s

    def _accept(self):
        self.s.gravity_mode = self.grav_mode.currentData()
        self.s.theta = self.theta.value()
        self.s.cooling = self.cooling.isChecked()
        self.s.u_floor = self.u_floor.value()
        self.s.t_cool = self.t_cool.value()
        self.s.sf_prob = self.sf_prob.value()
        self.s.du_sn = self.du_sn.value()
        self.accept()
