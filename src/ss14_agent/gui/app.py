"""Control window: start/pause the agent, see what it sees and why it chose what it did."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import playbook as playbooks
from ..config import Config


class ControlWindow(QMainWindow):
    def __init__(self, cfg: Config, playbook: str | None) -> None:
        super().__init__()
        self.cfg = cfg
        self.agent = None
        self._log_len = 0
        self.setWindowTitle("SS14 Agent")
        self.resize(1250, 780)

        # Top bar
        self.playbook = QComboBox()
        self.playbook.addItems(playbooks.available())
        self.playbook.setCurrentText(playbook or cfg.agent.playbook)
        self.note = QLineEdit()
        self.note.setPlaceholderText("Optional instruction, e.g. \"clean the hallway outside medbay\"")
        self.note.textChanged.connect(self._note_changed)
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._toggle_start)
        self.pause_btn = QPushButton("Pause (F8)")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.on_top = QCheckBox("Overlay")
        self.on_top.setToolTip(
            "Float this window above the game without taking focus. While the agent runs, "
            "clicks pass through it to the game; press F8 to pause and use the buttons."
        )
        self.on_top.toggled.connect(lambda _: self._apply_window_mode())
        self._mode: tuple[bool, bool] | None = None
        top = QHBoxLayout()
        top.addWidget(QLabel("Job:"))
        top.addWidget(self.playbook)
        top.addWidget(self.note, 1)
        top.addWidget(self.start_btn)
        top.addWidget(self.pause_btn)
        top.addWidget(self.on_top)

        # Live view
        self.view = QLabel("Press Start with SS14 open and joined to a server.")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(560, 360)
        self.view.setStyleSheet("background:#111;color:#aaa;")

        # Status + decision panel
        self.fields = {k: QLabel("–") for k in
                       ("Mode", "Window", "Loop", "Position", "Duty", "Doing", "Decision", "Chat / danger", "Latency")}
        form = QFormLayout()
        for k, lbl in self.fields.items():
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl.setWordWrap(True)
            form.addRow(f"{k}:", lbl)
        status_box = QGroupBox("Status")
        status_box.setLayout(form)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Option", "Probability"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        opts_box = QGroupBox("Top options at the last decision")
        ob = QVBoxLayout()
        ob.addWidget(self.table)
        opts_box.setLayout(ob)

        right = QVBoxLayout()
        right.addWidget(status_box)
        right.addWidget(opts_box, 1)
        right_w = QWidget()
        right_w.setLayout(right)

        upper = QSplitter(Qt.Horizontal)
        upper.addWidget(self.view)
        upper.addWidget(right_w)
        upper.setStretchFactor(0, 3)
        upper.setStretchFactor(1, 2)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Menlo", 11))
        self.log.setMaximumBlockCount(2000)

        split = QSplitter(Qt.Vertical)
        split.addWidget(upper)
        split.addWidget(self.log)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addWidget(split, 1)
        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)

    # ---- actions --------------------------------------------------------

    def _toggle_start(self) -> None:
        if self.agent is None:
            try:
                from ..agent import Agent
                from ..platform import get_platform

                self.agent = Agent(self.cfg, get_platform(self.cfg.capture.backend), self.playbook.currentText())
                self.agent.operator_note = self.note.text()
                self.agent.start()
            except Exception as exc:
                self.agent = None
                QMessageBox.critical(self, "Couldn't start", str(exc))
                return
            self.start_btn.setText("Stop")
            self.pause_btn.setEnabled(True)
            self.playbook.setEnabled(False)
        else:
            self.agent.stop()
            self._refresh()
            self.agent = None
            self.start_btn.setText("Start")
            self.pause_btn.setEnabled(False)
            self.playbook.setEnabled(True)

    def _toggle_pause(self) -> None:
        if self.agent is None:
            return
        if self.agent.paused:
            self.agent.resume()
        else:
            self.agent.pause("Pause button")

    def _note_changed(self, text: str) -> None:
        if self.agent is not None:
            self.agent.operator_note = text

    def _apply_window_mode(self) -> None:
        """Overlay: stay above the game and never take keyboard focus from it.

        The agent's keys and clicks go to the focused app at the cursor, so the
        game must stay focused and this window must not catch the clicks. The
        agent captures the game window itself, so covering it doesn't matter.
        """
        overlay = self.on_top.isChecked()
        running = self.agent is not None and not self.agent.paused
        mode = (overlay, overlay and running)
        if mode == self._mode:
            return
        self._mode = mode
        self.setWindowFlag(Qt.WindowStaysOnTopHint, overlay)
        self.setWindowFlag(Qt.WindowDoesNotAcceptFocus, overlay)
        self.setWindowFlag(Qt.WindowTransparentForInput, mode[1])
        self.setWindowOpacity(0.85 if mode[1] else 1.0)
        self.show()

    # ---- refresh --------------------------------------------------------

    def _refresh(self) -> None:
        self._apply_window_mode()
        a = self.agent
        if a is None:
            return
        s = a.status
        self.pause_btn.setText("Resume (F8)" if a.paused else "Pause (F8)")
        self.fields["Mode"].setText(s.mode)
        self.fields["Window"].setText(s.window or "–")
        self.fields["Loop"].setText(f"{s.fps:.0f} fps")
        self.fields["Position"].setText(f"{s.pos[0]:+.1f}, {s.pos[1]:+.1f} tiles from start")
        self.fields["Duty"].setText(s.duty or "–")
        self.fields["Doing"].setText(s.skill or "–")
        d = s.decision
        if d is not None:
            if d.error:
                self.fields["Decision"].setText(f"error: {d.error[:200]}")
            else:
                self.fields["Decision"].setText(f"{d.action} ({d.confidence:.0%}), on {d.reason}")
            self.fields["Chat / danger"].setText(f"spoken to {d.spoken_to:.0%} · danger {d.danger:.0%}")
            self.fields["Latency"].setText(f"{d.latency_ms:.0f} ms")
            self.table.setRowCount(len(d.top))
            for i, (opt, p) in enumerate(d.top):
                self.table.setItem(i, 0, QTableWidgetItem(opt))
                self.table.setItem(i, 1, QTableWidgetItem(f"{p:.1%}"))
        img = a.preview
        if img is not None:
            rgb = img.convert("RGB")
            data = rgb.tobytes()
            qimg = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            self.view.setPixmap(pix.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if s.log_count != self._log_len:
            self.log.setPlainText("\n".join(s.log))
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
            self._log_len = s.log_count

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self.agent is not None:
            self.agent.stop()
        super().closeEvent(event)


def run_gui(cfg: Config, playbook: str | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    win = ControlWindow(cfg, playbook)
    win.show()
    return app.exec()
