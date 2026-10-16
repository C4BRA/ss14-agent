"""One row of the control window per model role: provider, model, API key and a connection light."""

from __future__ import annotations

import threading
from collections.abc import Callable

import httpx
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from .. import keystore, models, providers

GREY, RED, GREEN = "#8a8a8a", "#e0453a", "#34c759"


class ProviderRow(QWidget):
    """Pick a provider and model, paste a key, and see whether the connection works.

    The light is grey until a check has run (or while one is running), red when the
    last check or call failed, and green when it succeeded. Network work runs on a
    background thread; `poll()` (called from the window's timer) applies results.
    """

    def __init__(self, role: str, cfg_section, catalog: dict[str, providers.Provider],
                 can_disable: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.role, self.cfg, self.catalog = role, cfg_section, catalog
        self._pending: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._check_id = 0

        self.enabled = QCheckBox(role.capitalize())
        self.enabled.setChecked(getattr(cfg_section, "enabled", True))
        self.enabled.setVisible(can_disable)
        self.enabled.toggled.connect(self._enabled_changed)
        title = QLabel(f"{role.capitalize()}:")
        title.setVisible(not can_disable)

        self.provider = QComboBox()
        for p in catalog.values():
            self.provider.addItem(p.name, p.id)
        self.provider.setCurrentIndex(max(0, self.provider.findData(cfg_section.provider)))
        self.provider.currentIndexChanged.connect(self._provider_changed)

        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setMinimumContentsLength(26)
        self.model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.model.addItem(cfg_section.model)
        self.model.currentTextChanged.connect(self._model_changed)

        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setMinimumWidth(170)
        self.key.returnPressed.connect(self._save_key)
        self.save_btn = QPushButton("Save key")
        self.save_btn.clicked.connect(self._save_key)

        self.light = QLabel()
        self.light.setFixedSize(14, 14)
        self.status = QLabel()
        self.status.setMinimumWidth(110)
        self.test_btn = QPushButton("Test")
        self.test_btn.clicked.connect(self.refresh)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        for w in (self.enabled, title, self.provider):
            row.addWidget(w)
        row.addWidget(self.model, 2)
        row.addWidget(self.key, 1)
        for w in (self.save_btn, self.light, self.status, self.test_btn):
            row.addWidget(w)
        self._update_key_field()
        self.set_light(GREY, "not checked yet")

    # ---- state -------------------------------------------------------------

    @property
    def prov(self) -> providers.Provider:
        return self.catalog[self.provider.currentData()]

    def set_light(self, color: str, text: str) -> None:
        self.light.setStyleSheet(f"background:{color}; border-radius:7px;")
        label = {GREEN: "connected", RED: "failed", GREY: "not connected"}[color]
        self.status.setText(label)
        self.light.setToolTip(text)
        self.status.setToolTip(text)

    def _later(self, fn: Callable[[], None]) -> None:
        with self._lock:
            self._pending.append(fn)

    def poll(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
        for fn in pending:
            fn()

    def _update_key_field(self) -> None:
        p = self.prov
        self.key.clear()
        self.key.setEnabled(p.needs_key)
        self.save_btn.setEnabled(p.needs_key)
        if not p.needs_key:
            self.key.setPlaceholderText("no key needed")
        elif p.key():
            self.key.setPlaceholderText("key saved (paste to replace)")
        else:
            self.key.setPlaceholderText(f"paste {p.name} API key")
        self.key.setToolTip(f"Get a key at {p.key_url}" if p.key_url else "")

    # ---- events ------------------------------------------------------------

    def _enabled_changed(self, on: bool) -> None:
        self.cfg.enabled = on
        for w in (self.provider, self.model, self.key, self.save_btn, self.test_btn):
            w.setEnabled(on and (w not in (self.key, self.save_btn) or self.prov.needs_key))
        if on:
            self.refresh()
        else:
            self.set_light(GREY, "turned off")

    def _provider_changed(self) -> None:
        p = self.prov
        self.cfg.provider = p.id
        self.cfg.endpoint = ""
        self.cfg.api_key_env = ""
        if hasattr(self.cfg, "image_mode"):
            self.cfg.image_mode = ""
        self.model.blockSignals(True)
        self.model.clear()
        if p.default_model:
            self.model.addItem(p.default_model)
        self.model.blockSignals(False)
        self.cfg.model = p.default_model
        self._update_key_field()
        self.refresh()

    def _model_changed(self, text: str) -> None:
        if text.strip():
            self.cfg.model = text.strip()   # clients read this on every call

    def _save_key(self) -> None:
        text = self.key.text().strip()
        if not text or not self.prov.needs_key:
            return
        where = keystore.save_key(self.prov.key_env, text)
        self._update_key_field()
        self.key.setPlaceholderText(f"key saved in {where}")
        self.refresh()

    # ---- background work ---------------------------------------------------

    def refresh(self) -> None:
        """Check the connection and reload the model list."""
        if hasattr(self.cfg, "enabled") and not self.cfg.enabled:
            return
        p = self.prov
        if p.needs_key and not p.key():
            self.set_light(GREY, f"No {p.name} key yet: paste one and press Save key.")
            return
        self._check_id += 1
        check_id = self._check_id
        self.set_light(GREY, "checking…")
        self.status.setText("checking…")
        model = self.cfg.model

        def work() -> None:
            try:
                note = providers.check(p, p.key(), model)
                result = (GREEN, note)
            except providers.ConnectionFailed as exc:
                result = (RED, str(exc))
            try:
                ids = models.models_for(self.role, self.cfg)
            except (providers.ConnectionFailed, httpx.HTTPError, ValueError, KeyError):
                # The list is a convenience; the light already says what's wrong.
                ids = None

            def apply() -> None:
                if check_id != self._check_id:
                    return  # a newer check superseded this one
                self.set_light(*result)
                if ids:
                    current = self.model.currentText()
                    self.model.blockSignals(True)
                    self.model.clear()
                    self.model.addItems(ids)
                    self.model.setCurrentText(current)
                    self.model.blockSignals(False)

            self._later(apply)

        threading.Thread(target=work, daemon=True).start()

    def report(self, ok: bool, detail: str) -> None:
        """Called with the outcome of real calls while the agent runs."""
        if ok:
            if self.status.text() != "connected":
                self.set_light(GREEN, detail)
        else:
            self.set_light(RED, detail)

