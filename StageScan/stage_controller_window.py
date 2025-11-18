from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from PySide6 import QtCore, QtWidgets

from Drivers.stage_driver import (
    NewportXPSStageDriver,
    StageConnectionSettings,
    StageDriverError,
)


@dataclass
class AxisControl:
    combo: QtWidgets.QComboBox
    position_label: QtWidgets.QLabel
    zero_label: QtWidgets.QLabel
    target_spin: QtWidgets.QDoubleSpinBox
    step_spin: QtWidgets.QDoubleSpinBox
    move_btn: QtWidgets.QPushButton
    jog_minus: QtWidgets.QPushButton
    jog_plus: QtWidgets.QPushButton
    zero_btn: QtWidgets.QPushButton
    goto_zero_btn: QtWidgets.QPushButton


class StageControllerWindow(QtWidgets.QMainWindow):
    """UI that configures and manually drives the Newport XPS stage."""

    _AXIS_ORDER = [
        ("x", "X axis", "mm", (-1000.0, 1000.0), 0.5, 5.0),
        ("y", "Y axis", "mm", (-1000.0, 1000.0), 0.5, 5.0),
        ("theta", "Rotation", "deg", (-720.0, 720.0), 1.0, 15.0),
    ]

    def __init__(self, parent, driver: NewportXPSStageDriver) -> None:
        super().__init__(parent=parent)
        self._driver = driver
        self._stage_table: Optional[QtWidgets.QTreeWidget] = None
        self._axis_controls: Dict[str, AxisControl] = {}
        self._build_ui()
        self._position_timer = QtCore.QTimer(self)
        self._position_timer.setInterval(250)
        self._position_timer.timeout.connect(self._update_axis_positions)
        self._position_timer.start()
        self._update_axis_positions()

    # ------------------------------------------------------------------ UI helpers
    def _build_ui(self) -> None:
        self.setWindowTitle("Stage Controller")
        self.resize(900, 600)
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        side_panel = QtWidgets.QVBoxLayout()
        side_panel.addWidget(self._build_connection_group())
        side_panel.addWidget(self._build_stage_overview_group(), 1)
        layout.addLayout(side_panel, 0)

        axes_panel = QtWidgets.QVBoxLayout()
        for axis_key, axis_label, units, limits, single_step, step_range in self._AXIS_ORDER:
            axes_panel.addWidget(
                self._build_axis_control(axis_key, axis_label, units, limits, single_step, step_range)
            )
        axes_panel.addStretch(1)
        layout.addLayout(axes_panel, 1)
        self._update_axis_assignments()

    def _build_connection_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Controller Connection")
        form = QtWidgets.QFormLayout(box)
        settings = self._driver.connection_settings()

        self.host_edit = QtWidgets.QLineEdit(settings.host)
        form.addRow("Host:", self.host_edit)

        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(settings.port)
        form.addRow("Port:", self.port_spin)

        self.user_edit = QtWidgets.QLineEdit(settings.username)
        form.addRow("Username:", self.user_edit)

        self.password_edit = QtWidgets.QLineEdit(settings.password)
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        form.addRow("Password:", self.password_edit)

        self.timeout_spin = QtWidgets.QDoubleSpinBox()
        self.timeout_spin.setRange(1.0, 120.0)
        self.timeout_spin.setDecimals(1)
        self.timeout_spin.setValue(settings.timeout_s)
        form.addRow("Timeout (s):", self.timeout_spin)

        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        form.addRow(self.btn_connect)

        self.connection_status = QtWidgets.QLabel("Controller disconnected")
        self.connection_status.setWordWrap(True)
        form.addRow(self.connection_status)
        return box

    def _build_stage_overview_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Controller Stages")
        layout = QtWidgets.QVBoxLayout(box)
        self._stage_table = QtWidgets.QTreeWidget()
        self._stage_table.setColumnCount(3)
        self._stage_table.setHeaderLabels(["Stage", "Type", "Group"])
        self._stage_table.setRootIsDecorated(False)
        layout.addWidget(self._stage_table, 1)
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_refresh_stage = QtWidgets.QPushButton("Refresh")
        self.btn_refresh_stage.clicked.connect(self._refresh_stage_list)
        btn_row.addWidget(self.btn_refresh_stage)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        self._refresh_stage_list()
        return box

    def _build_axis_control(
        self,
        axis_key: str,
        axis_label: str,
        units: str,
        limits: tuple[float, float],
        single_step: float,
        step_range: float,
    ) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(axis_label)
        layout = QtWidgets.QGridLayout(box)
        layout.setColumnStretch(1, 1)

        combo = QtWidgets.QComboBox()
        layout.addWidget(QtWidgets.QLabel("Stage:"), 0, 0)
        layout.addWidget(combo, 0, 1, 1, 3)

        position_label = QtWidgets.QLabel("Position: --")
        layout.addWidget(position_label, 1, 0, 1, 2)
        zero_label = QtWidgets.QLabel("Zero: --")
        layout.addWidget(zero_label, 1, 2, 1, 2)

        target_spin = QtWidgets.QDoubleSpinBox()
        target_spin.setRange(*limits)
        target_spin.setDecimals(3)
        target_spin.setSingleStep(single_step)
        layout.addWidget(QtWidgets.QLabel(f"Move to ({units}):"), 2, 0)
        layout.addWidget(target_spin, 2, 1)
        move_btn = QtWidgets.QPushButton("Move")
        layout.addWidget(move_btn, 2, 2)

        goto_zero_btn = QtWidgets.QPushButton("Go to zero")
        layout.addWidget(goto_zero_btn, 2, 3)

        step_spin = QtWidgets.QDoubleSpinBox()
        step_spin.setDecimals(3)
        step_spin.setRange(0.001, step_range)
        step_spin.setSingleStep(single_step)
        step_spin.setValue(single_step)
        layout.addWidget(QtWidgets.QLabel(f"Jog step ({units}):"), 3, 0)
        layout.addWidget(step_spin, 3, 1)
        jog_minus = QtWidgets.QPushButton("- Step")
        jog_plus = QtWidgets.QPushButton("+ Step")
        layout.addWidget(jog_minus, 3, 2)
        layout.addWidget(jog_plus, 3, 3)

        zero_btn = QtWidgets.QPushButton("Set zero here")
        layout.addWidget(zero_btn, 4, 0, 1, 4)

        controls = AxisControl(
            combo=combo,
            position_label=position_label,
            zero_label=zero_label,
            target_spin=target_spin,
            step_spin=step_spin,
            move_btn=move_btn,
            jog_minus=jog_minus,
            jog_plus=jog_plus,
            zero_btn=zero_btn,
            goto_zero_btn=goto_zero_btn,
        )
        self._axis_controls[axis_key] = controls

        combo.currentIndexChanged.connect(lambda _idx, axis=axis_key: self._on_axis_selection(axis))
        move_btn.clicked.connect(lambda _=False, axis=axis_key: self._move_axis(axis))
        jog_minus.clicked.connect(lambda _=False, axis=axis_key: self._jog_axis(axis, negative=True))
        jog_plus.clicked.connect(lambda _=False, axis=axis_key: self._jog_axis(axis, negative=False))
        zero_btn.clicked.connect(lambda _=False, axis=axis_key: self._zero_axis(axis))
        goto_zero_btn.clicked.connect(lambda _=False, axis=axis_key: self._goto_zero(axis))
        return box

    # ------------------------------------------------------------------ actions
    def _on_connect_clicked(self) -> None:
        if self._driver.is_connected():
            self._driver.disconnect()
            self.connection_status.setText("Controller disconnected")
            self.btn_connect.setText("Connect")
            if self._stage_table:
                self._stage_table.clear()
            self._update_axis_assignments()
            return
        settings = StageConnectionSettings(
            host=self.host_edit.text().strip() or "localhost",
            port=int(self.port_spin.value()),
            username=self.user_edit.text().strip() or "Administrator",
            password=self.password_edit.text(),
            timeout_s=float(self.timeout_spin.value()),
        )
        try:
            self._driver.connect(settings)
        except StageDriverError as exc:
            QtWidgets.QMessageBox.critical(self, "Stage Controller", str(exc))
            return
        self.btn_connect.setText("Disconnect")
        self.connection_status.setText("Controller connected. Refreshing stage list…")
        self._refresh_stage_list()
        self._update_axis_assignments()

    def _refresh_stage_list(self) -> None:
        if not self._stage_table:
            return
        self._stage_table.clear()
        if not self._driver.is_connected():
            return
        for info in self._driver.describe_stages():
            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, info.get("name", ""))
            item.setText(1, info.get("type", ""))
            item.setText(2, info.get("group", ""))
            self._stage_table.addTopLevelItem(item)
        self._update_axis_assignments()
        summary = self._driver.status_summary()
        self.connection_status.setText(summary)

    def _update_axis_assignments(self) -> None:
        names = self._driver.available_stage_names()
        for axis, controls in self._axis_controls.items():
            self._populate_axis_combo(axis, controls.combo, names)
        self._update_axis_positions()

    def _populate_axis_combo(self, axis: str, combo: QtWidgets.QComboBox, names: list[str]) -> None:
        current = self._driver.axis_assignment(axis)
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(not assigned)", None)
        for name in names:
            combo.addItem(name, name)
        if current:
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _on_axis_selection(self, axis: str) -> None:
        controls = self._axis_controls[axis]
        stage_name = controls.combo.currentData()
        try:
            self._driver.assign_axis(axis, stage_name)
        except StageDriverError as exc:
            QtWidgets.QMessageBox.critical(self, "Stage Controller", str(exc))
            self._update_axis_assignments()
            return
        self._update_axis_positions()
        self.connection_status.setText(self._driver.status_summary())

    def _move_axis(self, axis: str) -> None:
        controls = self._axis_controls[axis]
        try:
            self._driver.move_axis_to(axis, float(controls.target_spin.value()))
        except Exception as exc:  # pragma: no cover - hardware specific
            QtWidgets.QMessageBox.critical(self, "Stage Controller", str(exc))
        finally:
            self._update_axis_positions()

    def _jog_axis(self, axis: str, *, negative: bool) -> None:
        controls = self._axis_controls[axis]
        delta = float(controls.step_spin.value())
        if negative:
            delta = -delta
        try:
            self._driver.jog_axis(axis, delta)
        except Exception as exc:  # pragma: no cover - hardware specific
            QtWidgets.QMessageBox.critical(self, "Stage Controller", str(exc))
        finally:
            self._update_axis_positions()

    def _zero_axis(self, axis: str) -> None:
        try:
            self._driver.zero_axis(axis)
        except Exception as exc:  # pragma: no cover - hardware specific
            QtWidgets.QMessageBox.critical(self, "Stage Controller", str(exc))
        finally:
            self._update_axis_positions()
            self.connection_status.setText(self._driver.status_summary())

    def _goto_zero(self, axis: str) -> None:
        try:
            self._driver.move_axis_to(axis, 0.0)
        except Exception as exc:  # pragma: no cover - hardware specific
            QtWidgets.QMessageBox.critical(self, "Stage Controller", str(exc))
        finally:
            self._update_axis_positions()

    # ------------------------------------------------------------------ updates
    def _update_axis_positions(self) -> None:
        for axis, controls in self._axis_controls.items():
            try:
                value = self._driver.axis_position(axis)
            except Exception:
                value = None
            units = self._driver.axis_units(axis)
            if value is None:
                controls.position_label.setText("Position: --")
            else:
                controls.position_label.setText(f"Position: {value:.3f} {units}")
            try:
                zero = self._driver.axis_zero_reference(axis)
            except Exception:
                zero = None
            if zero is None:
                controls.zero_label.setText("Zero: --")
            else:
                controls.zero_label.setText(f"Zero: {zero:.3f} {units} absolute")
        if self._driver.is_connected():
            self.connection_status.setText(self._driver.status_summary())
