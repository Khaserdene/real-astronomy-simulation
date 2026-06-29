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
        self.grav_mode.addItem("Direct N\u00b2 (exact)", "direct")
        self.grav_mode.addItem("Barnes-Hut tree (fast, large N)", "bh")
        self.grav_mode.setCurrentIndex(0 if self.s.gravity_mode == "direct" else 1)
        self.theta = self._spin(0.1, 1.5, 0.05, 3, self.s.theta)
        gf.addRow("Solver", self.grav_mode)
        gf.addRow("BH opening angle \u03b8", self.theta)
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

        # --- star formation ---
        sfg = QGroupBox("Star formation (Schmidt law)")
        sff = QFormLayout(sfg)
        self.eps_ff = self._spin(0.001, 0.2, 0.005, 3, self.s.eps_ff)
        self.eps_ff.setToolTip(
            "SF efficiency per free-fall time (\u03b5_ff). "
            "Physical range: 0.01\u20130.02. Higher = faster star formation.")
        self.sf_density_factor = self._spin(1.0, 50.0, 1.0, 1,
                                            self.s.sf_density_factor)
        self.sf_density_factor.setToolTip(
            "Density threshold = factor \u00d7 median gas density. "
            "Gas denser than this can form stars.")
        sff.addRow("SF efficiency \u03b5_ff", self.eps_ff)
        sff.addRow("Density threshold factor", self.sf_density_factor)
        v.addWidget(sfg)

        # --- supernova feedback ---
        s = QGroupBox("Supernova feedback")
        sf = QFormLayout(s)
        self.du_sn = self._spin(0.0, 5000.0, 50.0, 1, self.s.du_sn)
        self.du_sn.setToolTip(
            "Thermal energy budget per SN event, split among neighbors. "
            "N-scaled by particle mass.")
        self.v_sn = self._spin(0.0, 500.0, 10.0, 1, self.s.v_sn)
        self.v_sn.setToolTip(
            "Kinetic kick velocity per SN event (km/s). "
            "Drives outflows and gas fountains.")
        self.t_sn_max = self._spin(0.005, 0.5, 0.005, 3, self.s.t_sn_max)
        self.t_sn_max.setToolTip(
            "Max stellar lifetime for SN-capable stars (~49 Myr at 0.05). "
            "Only the ~5% massive stars with lifetime < this will fire SN.")
        self.r_fb = self._spin(0.1, 5.0, 0.1, 2, self.s.r_fb)
        self.r_fb.setToolTip(
            "Feedback radius (kpc). SN energy reaches gas within this distance.")
        self.f_return = self._spin(0.0, 0.8, 0.05, 2, self.s.f_return)
        self.f_return.setToolTip(
            "Mass return fraction: how much of the exploding star's mass "
            "is returned to the ISM. Remnant keeps 1 - f_return.")
        sf.addRow("SN energy du_sn", self.du_sn)
        sf.addRow("SN kick velocity v_sn", self.v_sn)
        sf.addRow("SN max lifetime t_sn_max", self.t_sn_max)
        sf.addRow("Feedback radius r_fb", self.r_fb)
        sf.addRow("Mass return fraction", self.f_return)
        v.addWidget(s)

        # --- dynamic N / baryon cycle ---
        bc = QGroupBox("Baryon cycle (dynamic particle count)")
        bcf = QFormLayout(bc)
        self.dynamic_baryons = QCheckBox("Enabled (SN→gas, hypernova→BH, "
                                         "escaper removal)")
        self.dynamic_baryons.setChecked(bool(getattr(self.s, "dynamic_baryons", True)))
        self.sn_gas_split = self._spin(1, 16, 1, 0,
                                       getattr(self.s, "sn_gas_split", 4))
        self.sn_gas_split.setToolTip("Gas particles spawned when a star explodes "
                                     "(one supernova → many gas particles).")
        self.gas_inflow = QCheckBox("Gas inflow (replenish ejected particles)")
        self.gas_inflow.setChecked(bool(getattr(self.s, "gas_inflow", True)))
        self.inflow_radius = self._spin(5.0, 200.0, 5.0, 1,
                                        getattr(self.s, "inflow_radius", 40.0))
        self.inflow_radius.setToolTip("Outer shell radius (kpc) where fresh "
                                      "infalling gas appears.")
        self.escape_radius = self._spin(10.0, 300.0, 10.0, 1,
                                        getattr(self.s, "escape_radius", 60.0))
        self.escape_radius.setToolTip("Unbound particles beyond this radius (kpc) "
                                      "are removed (live-gravity scenarios only).")
        self.rebuild_every = self._spin(1, 200, 1, 0,
                                        getattr(self.s, "rebuild_every", 20))
        self.rebuild_every.setToolTip("Steps between dynamic-N rebuilds. Smaller = "
                                      "more responsive births/deaths, more overhead.")
        bcf.addRow(self.dynamic_baryons)
        bcf.addRow("Gas per supernova", self.sn_gas_split)
        bcf.addRow(self.gas_inflow)
        bcf.addRow("Inflow radius", self.inflow_radius)
        bcf.addRow("Escape radius", self.escape_radius)
        bcf.addRow("Rebuild every (steps)", self.rebuild_every)
        v.addWidget(bc)

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
        self.s.eps_ff = self.eps_ff.value()
        self.s.sf_density_factor = self.sf_density_factor.value()
        self.s.du_sn = self.du_sn.value()
        self.s.v_sn = self.v_sn.value()
        self.s.t_sn_max = self.t_sn_max.value()
        self.s.r_fb = self.r_fb.value()
        self.s.f_return = self.f_return.value()
        self.s.dynamic_baryons = self.dynamic_baryons.isChecked()
        self.s.sn_gas_split = int(self.sn_gas_split.value())
        self.s.gas_inflow = self.gas_inflow.isChecked()
        self.s.inflow_radius = self.inflow_radius.value()
        self.s.escape_radius = self.escape_radius.value()
        self.s.rebuild_every = int(self.rebuild_every.value())
        self.accept()
