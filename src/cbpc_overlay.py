"""Semi-transparent physics control panel anchored to the bottom-right of the 3D viewport."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

STEPS_PER_UNIT = 30  # slider resolution: 1.0 (More Gravity) == 30
SOFTNESS_STEPS = 20  # slider resolution: 1.0x == 20


class CbpcOverlay(QFrame):
    """SMP and CBPC enable, breast gravity, and per-area softness controls."""

    enabledChanged = Signal(bool)
    smpEnabledChanged = Signal(bool)
    gravityChanged = Signal(float)
    softnessChanged = Signal(str, float)

    MARGIN = 12

    def __init__(
        self,
        parent: QWidget,
        minimum: float,
        maximum: float,
        softness_range: Tuple[float, float] = (0.0, 5.0),
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cbpcOverlay")
        self.setStyleSheet(
            "#cbpcOverlay { background-color: rgba(20, 22, 28, 170);"
            " border: 1px solid rgba(255, 255, 255, 50); border-radius: 6px; }"
            "#cbpcOverlay QCheckBox, #cbpcOverlay QLabel { color: #e6e6e6; }"
            "#cbpcOverlay QCheckBox:disabled, #cbpcOverlay QLabel:disabled { color: #8a8a8a; }"
        )
        self.captions: dict[str, QLabel] = {}

        self.smp_checkbox = QCheckBox("Enable SMP Physics", self)
        self._smp_tooltip = ""
        self._smp_unavailable_tip = ""
        self.checkbox = QCheckBox("Enable CBPC Physics", self)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(round(minimum * STEPS_PER_UNIT), round(maximum * STEPS_PER_UNIT))
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(STEPS_PER_UNIT // 3)
        self.slider.setMinimumWidth(140)
        self.label = QLabel(self)
        self.softness_sliders: dict[str, QSlider] = {}
        self.softness_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addWidget(self.smp_checkbox)
        layout.addWidget(self.checkbox)
        layout.addLayout(self._row("gravity", "Gravity", self.slider, self.label))
        for group, caption in (("breasts", "Breasts"), ("butt", "Butt"), ("belly", "Belly")):
            slider = QSlider(Qt.Orientation.Horizontal, self)
            slider.setRange(round(softness_range[0] * SOFTNESS_STEPS), round(softness_range[1] * SOFTNESS_STEPS))
            slider.setValue(SOFTNESS_STEPS)
            slider.setAccessibleName(f"{caption} softness")
            label = QLabel(self)
            self.softness_sliders[group] = slider
            self.softness_labels[group] = label
            layout.addLayout(self._row(group, f"{caption} softness", slider, label))
            slider.valueChanged.connect(lambda value, group=group: self._on_softness(group, value))

        self.smp_checkbox.toggled.connect(self.smpEnabledChanged)
        self.checkbox.toggled.connect(self.enabledChanged)
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
        self.softness_labels[group].setText(f"{self.softness_sliders[group].value() / SOFTNESS_STEPS:.2f}x")

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

    def sync(self, enabled: bool, available: bool, gravity: float, bone_count: int = 0,
             softness: Optional[Mapping[str, float]] = None) -> None:
        """Reflect manager state without emitting change signals."""
        controls = (self.checkbox, self.slider, *self.softness_sliders.values())
        for w in controls:
            w.blockSignals(True)
        self.checkbox.setEnabled(available)
        self.slider.setEnabled(available and enabled)
        self.label.setEnabled(available and enabled)
        for group, slider in self.softness_sliders.items():
            slider.setEnabled(available and enabled)
            self.softness_labels[group].setEnabled(available and enabled)
            slider.setValue(round((softness[group] if softness is not None else 1.0) * SOFTNESS_STEPS))
        for c in self.captions.values():
            c.setEnabled(available and enabled)
        self.checkbox.setChecked(bool(enabled and available))
        self.slider.setValue(round(gravity * STEPS_PER_UNIT))
        for w in controls:
            w.blockSignals(False)
        self._update_label()
        for group in self.softness_sliders:
            self._update_softness_label(group)

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        self.move(
            parent.width() - self.width() - self.MARGIN,
            parent.height() - self.height() - self.MARGIN,
        )
        self.raise_()
