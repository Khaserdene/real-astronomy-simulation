"""Scene builder dialog -- parameter-based object composition.

Drop objects into a scene, give each a position, an initial (bulk) velocity and
an optional spin, and build them into one runnable scene: two galaxies set on a
collision course, two planets aimed at each other, and so on.  This is the
parameter half of the editor (the 3D brush is the interactive half); both feed
the same :class:`~core.scene_spec.SceneSpec` consumed by ``SimController``.

Objects in one scene share an engine kind (v1), so the template choices are
filtered by domain (galaxy vs planetary) and the dialog refuses to mix kinds.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QListWidget, QLabel, QWidget,
    QDialogButtonBox, QMessageBox,
)

from core.objects import OBJECTS, templates_for, compose_spec
from core.scene_spec import ObjectSpec

_DOMAINS = ["galaxy", "planetary"]


class SceneBuilderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scene builder")
        self.resize(560, 460)
        self._objects: list[ObjectSpec] = []
        self._loading = False  # guard form<->model echo

        root = QHBoxLayout(self)

        # --- left: object list + add/remove ---
        left = QVBoxLayout()
        self.domain_combo = QComboBox()
        self.domain_combo.addItems(_DOMAINS)
        self.domain_combo.currentTextChanged.connect(self._on_domain)
        left.addWidget(QLabel("Domain"))
        left.addWidget(self.domain_combo)
        self.add_combo = QComboBox()
        left.addWidget(QLabel("Add object"))
        left.addWidget(self.add_combo)
        self.add_btn = QPushButton("+ Add")
        self.add_btn.clicked.connect(self._add_object)
        left.addWidget(self.add_btn)
        self.obj_list = QListWidget()
        self.obj_list.currentRowChanged.connect(self._load_form)
        left.addWidget(self.obj_list, 1)
        self.remove_btn = QPushButton("− Remove")
        self.remove_btn.clicked.connect(self._remove_object)
        left.addWidget(self.remove_btn)
        root.addLayout(left, 0)

        # --- right: per-object editor ---
        self.editor = self._build_editor()
        root.addWidget(self.editor, 1)

        self._on_domain(self.domain_combo.currentText())

        # OK / Cancel under the editor (re-parent into right column).
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        self.editor.layout().addWidget(bb)

    # --------------------------------------------------------------- editor UI
    def _build_editor(self) -> QWidget:
        box = QGroupBox("Selected object")
        outer = QVBoxLayout(box)
        form = QFormLayout()

        self.template_combo = QComboBox()
        self.template_combo.currentTextChanged.connect(self._write_back)
        self.n_spin = QSpinBox(); self.n_spin.setRange(10, 2_000_000)
        self.n_spin.setGroupSeparatorShown(True); self.n_spin.setSingleStep(1000)
        self.n_spin.valueChanged.connect(self._write_back)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 9999)
        self.seed_spin.valueChanged.connect(self._write_back)
        form.addRow("Template", self.template_combo)
        form.addRow("Particles", self.n_spin)
        form.addRow("Seed", self.seed_spin)

        self.pos = [self._coord() for _ in range(3)]
        self.vel = [self._coord() for _ in range(3)]
        form.addRow("Position (kpc)", _xyz_row(self.pos))
        form.addRow("Velocity (km/s)", _xyz_row(self.vel))

        self.spin_spin = QDoubleSpinBox()
        self.spin_spin.setRange(-50.0, 50.0); self.spin_spin.setSingleStep(0.1)
        self.spin_spin.setToolTip("Solid-body spin about z (km/s per kpc)")
        self.spin_spin.valueChanged.connect(self._write_back)
        form.addRow("Spin (z)", self.spin_spin)

        outer.addLayout(form)
        self.params_label = QLabel("")
        self.params_label.setWordWrap(True)
        self.params_label.setStyleSheet("color: gray; font-size: 10px;")
        outer.addWidget(self.params_label)
        outer.addStretch(1)
        box.setEnabled(False)   # nothing selected yet
        return box

    def _coord(self) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-500.0, 500.0)
        s.setDecimals(1)
        s.setSingleStep(1.0)
        s.valueChanged.connect(self._write_back)
        return s

    # --------------------------------------------------------------- behaviour
    def _on_domain(self, domain: str):
        self.add_combo.clear()
        for name, t in templates_for(domain).items():
            self.add_combo.addItem(t.label, name)
        # Templates allowed for editing the selected object follow the domain too.
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        for name, t in templates_for(domain).items():
            self.template_combo.addItem(t.label, name)
        self.template_combo.blockSignals(False)
        if self._objects:
            self._objects.clear()
            self.obj_list.clear()
            self._set_editor_enabled(False)

    def _add_object(self):
        name = self.add_combo.currentData()
        if not name:
            return
        self._objects.append(ObjectSpec(template=name, n=20000, seed=0))
        self.obj_list.addItem(self._summary(self._objects[-1]))
        self.obj_list.setCurrentRow(len(self._objects) - 1)

    def _remove_object(self):
        r = self.obj_list.currentRow()
        if 0 <= r < len(self._objects):
            self._objects.pop(r)
            self.obj_list.takeItem(r)
            self._set_editor_enabled(bool(self._objects))

    def _load_form(self, row: int):
        if not (0 <= row < len(self._objects)):
            self._set_editor_enabled(False)
            return
        o = self._objects[row]
        self._loading = True
        self._set_editor_enabled(True)
        i = self.template_combo.findData(o.template)
        if i >= 0:
            self.template_combo.setCurrentIndex(i)
        self.n_spin.setValue(o.n)
        self.seed_spin.setValue(o.seed)
        for k in range(3):
            self.pos[k].setValue(float(o.position[k]))
            self.vel[k].setValue(float(o.velocity[k]))
        self.spin_spin.setValue(float(o.spin or 0.0))
        self.params_label.setText(
            "defaults: " + ", ".join(f"{k}={v}" for k, v in
                                     OBJECTS[o.template].defaults.items()))
        self._loading = False

    def _write_back(self, *_):
        row = self.obj_list.currentRow()
        if self._loading or not (0 <= row < len(self._objects)):
            return
        o = self._objects[row]
        o.template = self.template_combo.currentData() or o.template
        o.n = self.n_spin.value()
        o.seed = self.seed_spin.value()
        o.position = [c.value() for c in self.pos]
        o.velocity = [c.value() for c in self.vel]
        o.spin = self.spin_spin.value() or None
        self.obj_list.item(row).setText(self._summary(o))
        self.params_label.setText(
            "defaults: " + ", ".join(f"{k}={v}" for k, v in
                                     OBJECTS[o.template].defaults.items()))

    def _accept(self):
        if not self._objects:
            QMessageBox.information(self, "Empty scene", "Add at least one object.")
            return
        self.accept()

    def result_spec(self):
        """The composed SceneSpec (call only after exec() returned Accepted)."""
        return compose_spec(self._objects, domain=self.domain_combo.currentText())

    # ------------------------------------------------------------------ helpers
    def _set_editor_enabled(self, on: bool):
        self.editor.setEnabled(on)

    @staticmethod
    def _summary(o: ObjectSpec) -> str:
        pos = ",".join(f"{v:g}" for v in o.position)
        return f"{o.template}  N={o.n:,}  @({pos})"


def _xyz_row(spins) -> QWidget:
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    for s in spins:
        row.addWidget(s)
    return w
