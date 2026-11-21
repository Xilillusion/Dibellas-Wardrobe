"""Physics controls for the settings sidebar."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from .config import SOFTNESS_MAX, SOFTNESS_MIN

STEPS_PER_UNIT = 30  # slider resolution: 1.0 (More Gravity) == 30
SOFTNESS_STEPS = 2  # slider resolution: 1.0x == 2


class CbpcOverlay(QFrame):
    """SMP enable, CBPC breast gravity, and per-area softness controls."""

    smpEnabledChanged = Signal(bool)
    gravityChanged = Signal(float)
    softnessChanged = Signal(str, float)

    def __init__(
        self,
        parent: QWidget,
        minimum: float,
        maximum: float,
        softness_range: Tuple[float, float] = (SOFTNESS_MIN, SOFTNESS_MAX),
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cbpcOverlay")
        self.captions: dict[str, QLabel] = {}

        self.smp_checkbox = QCheckBox("Enable SMP Physics", self)
        self._smp_tooltip = ""
        self._smp_unavailable_tip = ""
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(round(minimum * STEPS_PER_UNIT), round(maximum * STEPS_PER_UNIT))
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(STEPS_PER_UNIT // 3)
        self.slider.setMinimumWidth(140)
        self.label = QLabel(self)
        self.softness_sliders: dict[str, QSlider] = {}
        self.softness_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.smp_checkbox)
        layout.addLayout(self._row("gravity", "Gravity", self.slider, self.label))
        for group, caption in (("breasts", "Breasts"), ("butt", "Butt"), ("belly", "Belly"), ("thigh", "Thigh")):
            slider = QSlider(Qt.Orientation.Horizontal, self)
            slider.setRange(round(softness_range[0] * SOFTNESS_STEPS), round(softness_range[1] * SOFTNESS_STEPS))
            slider.setPageStep(1)
            slider.setValue(SOFTNESS_STEPS)
            slider.setAccessibleName(f"{caption} softness")
            label = QLabel(self)
            self.softness_sliders[group] = slider
            self.softness_labels[group] = label
            layout.addLayout(self._row(group, f"{caption} softness", slider, label))
            slider.valueChanged.connect(lambda value, group=group: self._on_softness(group, value))

        self.smp_checkbox.toggled.connect(self.smpEnabledChanged)
        self.slider.valueChanged.connect(self._on_slider)
        self._update_label()
        for group in self.softness_sliders:
            self._update_softness_label(group)
        self.adjustSize()

    def _row(self, key: str, caption: str, slider: QSlider, value: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        name = QLabel(caption, self)
        row.addWidget(name)
        row.addWidget(slider, 1)
        value.setMinimumWidth(38)
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(value)
        self.captions[key] = name
        return row

    def _update_label(self) -> None:
        factor = self.slider.value() / STEPS_PER_UNIT
        self.label.setText(f"{factor:.2f}x")

    def _update_softness_label(self, group: str) -> None:
        self.softness_labels[group].setText(f"{self.softness_sliders[group].value() / SOFTNESS_STEPS:.1f}x")

    def _on_softness(self, group: str, value: int) -> None:
        self._update_softness_label(group)
        self.softnessChanged.emit(group, value / SOFTNESS_STEPS)

    def _on_slider(self, value: int) -> None:
        self._update_label()
        self.gravityChanged.emit(value / STEPS_PER_UNIT)

    def set_smp_texts(self, label: str, tooltip: str) -> None:
        self.smp_checkbox.setText(label)
        self._smp_tooltip = tooltip
        self._apply_smp_tooltip()

    def set_smp_state(self, enabled: bool, available: bool, unavailable_tip: str = "") -> None:
        with QSignalBlocker(self.smp_checkbox):
            self.smp_checkbox.setChecked(bool(enabled))
        self._smp_unavailable_tip = "" if available else unavailable_tip
        self.smp_checkbox.setEnabled(bool(available))
        self._apply_smp_tooltip()

    def _apply_smp_tooltip(self) -> None:
        self.smp_checkbox.setToolTip(self._smp_unavailable_tip or self._smp_tooltip)

    def sync(self, available: bool, gravity: float,
             softness: Optional[Mapping[str, float]] = None) -> None:
        """Reflect manager state without emitting change signals."""
        controls = (self.slider, *self.softness_sliders.values())
        for w in controls:
            w.blockSignals(True)
        self.slider.setEnabled(available)
        self.label.setEnabled(available)
        for group, slider in self.softness_sliders.items():
            slider.setEnabled(available)
            self.softness_labels[group].setEnabled(available)
            slider.setValue(round((softness[group] if softness is not None else 1.0) * SOFTNESS_STEPS))
        for c in self.captions.values():
            c.setEnabled(available)
        self.slider.setValue(round(gravity * STEPS_PER_UNIT))
        for w in controls:
            w.blockSignals(False)
        self._update_label()
        for group in self.softness_sliders:
            if softness is None:
                self._update_softness_label(group)
            else:
                self.softness_labels[group].setText(f"{softness[group]:.1f}x")
