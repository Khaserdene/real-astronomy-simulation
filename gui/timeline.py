"""Bottom timeline -- a video-editor-style transport bar for saved snapshots.

The widget owns nothing about the simulation; it just tracks a list of frame
paths and a current index, and emits:

* ``frameSelected(i)`` -- the user scrubbed/stepped, or playback advanced, to
  saved frame ``i`` (the view should show that snapshot).
* ``goLive()``        -- playback ran off the end of the saved frames; the app
  should hand back to the live simulation (continue stepping).
* ``refreshRequested()`` -- the user asked to rescan the output folder.

While the live simulation is running and dropping new snapshots, the host calls
:meth:`set_frames` repeatedly; the bar quietly *follows* the newest frame
(updating the slider position without emitting ``frameSelected``) so it never
hijacks the live view -- until the user actually scrubs or presses play.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QSpinBox,
)


class TimelineWidget(QWidget):
    frameSelected = pyqtSignal(int)
    goLive = pyqtSignal()
    refreshRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames: list[str] = []
        self._playing = False
        self._follow = True      # track the newest frame until the user scrubs
        self._suppress = False   # set slider position without emitting

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)

        self.first_btn = _btn("⏮", "First frame", self._first)
        self.prev_btn = _btn("⏪", "Previous frame", self._prev)
        self.play_btn = _btn("▶", "Play / pause", self._toggle_play)
        self.next_btn = _btn("⏩", "Next frame", self._next)
        self.last_btn = _btn("⏭", "Last frame", self._last)
        self.live_btn = _btn("Live ⏵", "Continue the live simulation",
                             self.goLive.emit)
        self.live_btn.setFixedWidth(70)
        for b in (self.first_btn, self.prev_btn, self.play_btn,
                  self.next_btn, self.last_btn, self.live_btn):
            row.addWidget(b)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.setTracking(True)  # emit continuously while dragging (scrub)
        self.slider.valueChanged.connect(self._on_slider)
        row.addWidget(self.slider, 1)

        self.frame_label = QLabel("no frames")
        self.frame_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.frame_label.setMinimumWidth(170)
        row.addWidget(self.frame_label)

        row.addWidget(QLabel("fps"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(12)
        self.fps_spin.valueChanged.connect(self._apply_fps)
        row.addWidget(self.fps_spin)

        self.refresh_btn = _btn("⟳", "Rescan folder", self.refreshRequested.emit)
        row.addWidget(self.refresh_btn)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._apply_fps()
        self._set_controls_enabled(False)

    # --------------------------------------------------------------- public API
    def set_frames(self, paths) -> None:
        """Replace the frame list, keeping (or following) a sensible position."""
        cur = self.slider.value()
        self._frames = list(paths)
        n = len(self._frames)
        self._suppress = True
        self.slider.setRange(0, max(n - 1, 0))
        self._suppress = False
        self._set_controls_enabled(n > 0)
        if n == 0:
            self.frame_label.setText("no frames")
            return
        if self._follow and not self._playing:
            self._set_slider(n - 1)          # ride the newest frame, no emit
        else:
            self._set_slider(min(cur, n - 1))
        self._update_label()

    def current_index(self) -> int:
        return self.slider.value()

    def stop(self) -> None:
        self._set_playing(False)

    def follow_live(self) -> None:
        """Resume auto-following the newest frame (called when going live)."""
        self._follow = True

    # ---------------------------------------------------------------- transport
    def _first(self):
        self._scrub_to(0)

    def _prev(self):
        self._scrub_to(self.slider.value() - 1)

    def _next(self):
        self._scrub_to(self.slider.value() + 1)

    def _last(self):
        self._scrub_to(self.slider.maximum())

    def _scrub_to(self, i: int):
        if not self._frames:
            return
        self._follow = False
        self._set_slider(max(0, min(i, self.slider.maximum())), emit=True)

    def _toggle_play(self):
        if not self._frames:
            return
        if not self._playing:
            self._follow = False
            if self.slider.value() >= self.slider.maximum():
                self._set_slider(0, emit=True)   # replay from the start
            self._set_playing(True)
        else:
            self._set_playing(False)

    def _advance(self):
        i = self.slider.value()
        if i < self.slider.maximum():
            self._set_slider(i + 1, emit=True)
        else:
            self._set_playing(False)
            self._follow = True
            self.goLive.emit()

    # ------------------------------------------------------------------ helpers
    def _set_playing(self, on: bool):
        self._playing = on
        self.play_btn.setText("⏸" if on else "▶")
        if on:
            self._timer.start()
        else:
            self._timer.stop()

    def _apply_fps(self):
        self._timer.setInterval(int(1000 / max(self.fps_spin.value(), 1)))

    def _set_slider(self, i: int, emit: bool = False):
        if not emit:
            self._suppress = True
        self.slider.setValue(i)
        self._suppress = False
        if not emit:
            self._update_label()

    def _on_slider(self, val: int):
        if self._suppress:
            return
        self._follow = False  # any real slider move = user took control
        self._update_label()
        if self._frames:
            self.frameSelected.emit(val)

    def _update_label(self):
        n = len(self._frames)
        if n == 0:
            self.frame_label.setText("no frames")
            return
        i = self.slider.value()
        self.frame_label.setText(
            f"{i + 1} / {n}  {os.path.basename(self._frames[i])}")

    def _set_controls_enabled(self, on: bool):
        for b in (self.first_btn, self.prev_btn, self.play_btn,
                  self.next_btn, self.last_btn):
            b.setEnabled(on)
        self.slider.setEnabled(on)


def _btn(text, tip, slot) -> QPushButton:
    b = QPushButton(text)
    b.setToolTip(tip)
    b.setFixedWidth(36)
    b.clicked.connect(lambda _=False: slot())
    return b
