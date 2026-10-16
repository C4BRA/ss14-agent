"""Control window: start/pause the agent, see what it sees and why it chose what it did."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .. import playbook as playbooks
from ..config import Config
from ..providers import DECISION_PROVIDERS, ORCHESTRATOR_PROVIDERS
from .provider_row import ProviderRow


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
        self.sees = QComboBox()
        self.sees.addItem("Text (no image)", "text")
        self.sees.addItem("Screenshot", "image")
        self.sees.setCurrentIndex(max(0, self.sees.findData(cfg.perception.mode)))
        self.sees.setToolTip(
            "What the decision model gets. Text: a tile map and named things read from the screen by "
            "our own code. Screenshot: the annotated image. Takes effect on the next decision."
        )
        self.sees.currentIndexChanged.connect(lambda _: setattr(cfg.perception, "mode", self.sees.currentData()))
        top = QHBoxLayout()
        top.addWidget(QLabel("Job:"))
        top.addWidget(self.playbook)
        top.addWidget(self.note, 1)
        top.addWidget(QLabel("Model sees:"))
        top.addWidget(self.sees)
        top.addWidget(self.start_btn)
        top.addWidget(self.pause_btn)
        top.addWidget(self.on_top)

        # Provider, model, key and connection light for each model role.
        self.decision_row = ProviderRow("decision", cfg.decision, DECISION_PROVIDERS)
        self.orch_row = ProviderRow("orchestrator", cfg.orchestrator, ORCHESTRATOR_PROVIDERS, can_disable=True)

        # Live view
        self.view = QLabel("Press Start with SS14 open and joined to a server.")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(560, 360)
        self.view.setStyleSheet("background:#111;color:#aaa;")
        self.view.setToolTip("Labels under the boxes: green = read from a right-click menu, "
                             "amber = recognised from a learned picture, grey = unknown.")

        # What the decision model receives in text mode, and the raw state of the last call.
        self.scene_view = QPlainTextEdit()
        self.scene_view.setReadOnly(True)
        mono = QFont("Menlo", 12)
        mono.setStyleHint(QFont.Monospace)
        self.scene_view.setFont(mono)
        self.scene_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.scene_view.setPlaceholderText("The map and named things appear here once the agent is running.")
        self.request_view = QPlainTextEdit()
        self.request_view.setReadOnly(True)
        self.request_view.setFont(QFont("Menlo", 11))
        self.request_view.setPlaceholderText("The state sent with the last decision call appears here.")
        self.tabs = QTabWidget()
        self.tabs.addTab(self.view, "Screen")
        self.tabs.addTab(self.scene_view, "Model view")
        self.tabs.addTab(self.request_view, "Last request")
        self._scene_text = ""
        self._request_text = ""

        # Status + decision panel
        self.fields = {k: QLabel("–") for k in
                       ("Mode", "Window", "Loop", "Position", "Learned", "Goal", "Plan step", "Duty", "Doing", "Decision",
                        "Chat / danger", "Latency")}
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
        upper.addWidget(self.tabs)
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
        root.addWidget(self.decision_row)
        root.addWidget(self.orch_row)
        root.addWidget(split, 1)
        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)
        self.decision_row.refresh()
        self.orch_row.refresh()

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
            self.orch_row.enabled.setEnabled(False)  # on/off is fixed while running; models can change
        else:
            self.agent.stop()
            self._refresh()
            self.agent = None
            self.start_btn.setText("Start")
            self.pause_btn.setEnabled(False)
            self.playbook.setEnabled(True)
            self.orch_row.enabled.setEnabled(True)

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
        self.decision_row.poll()
        self.orch_row.poll()
        a = self.agent
        if a is None:
            return
        s = a.status
        self.pause_btn.setText("Resume (F8)" if a.paused else "Pause (F8)")
        self.fields["Mode"].setText(s.mode)
        self.fields["Window"].setText(s.window or "–")
        self.fields["Loop"].setText(f"{s.fps:.0f} fps")
        self.fields["Position"].setText(f"{s.pos[0]:+.1f}, {s.pos[1]:+.1f} tiles from start")
        lib = a.library
        self.fields["Learned"].setText(f"{len(lib.names)} things from {len(lib)} pictures")
        if s.scene_text != self._scene_text:
            self._scene_text = s.scene_text
            self.scene_view.setPlainText(s.scene_text)
        if s.request_text != self._request_text:
            self._request_text = s.request_text
            bar = self.request_view.verticalScrollBar()
            keep = bar.value()
            self.request_view.setPlainText(s.request_text)
            bar.setValue(keep)
        if s.plan is not None and not s.plan_error:
            self.orch_row.report(True, f"last plan {s.plan.latency_ms / 1000:.1f} s")
        if s.plan is not None:
            self.fields["Goal"].setText(f"{s.plan.goal}  ({s.plan.latency_ms / 1000:.1f} s)")
            self.fields["Plan step"].setText(
                " → ".join(f"[{st}]" if i == s.plan.current_step else st for i, st in enumerate(s.plan.steps))
            )
        elif s.plan_error:
            self.fields["Goal"].setText(f"planning error: {s.plan_error[:200]}")
            self.orch_row.report(False, s.plan_error[:300])
        elif a.orchestrator is None:
            self.fields["Goal"].setText("orchestrator off")
        self.fields["Duty"].setText(s.duty or "–")
        self.fields["Doing"].setText(s.skill or "–")
        d = s.decision
        if d is not None:
            if d.error:
                self.fields["Decision"].setText(f"error: {d.error[:200]}")
                self.decision_row.report(False, d.error[:300])
            else:
                self.decision_row.report(True, f"last call {d.latency_ms:.0f} ms")
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
