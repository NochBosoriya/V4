"""Settings tab: providers/keys, routing, usage, TTS extras, app settings,
model manager, and settings export/import. No licensing/activation UI."""
from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core import i18n
from app.core.config_manager import (
    FEATURES,
    PROVIDER_NAMES,
    PROVIDERS,
    Route,
    key_out_of_credit,
)
from app.core.i18n import get_language, tr
from app.core.logger import get_logger
from app.core.paths import models_dir

if TYPE_CHECKING:
    from app.ui.context import AppContext

log = get_logger("ui.settings_tab")

_DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "custom": "",
}
_PROVIDER_TITLES = {"deepseek": "DeepSeek", "openai": "OpenAI", "openrouter": "OpenRouter",
                    "gemini": "Gemini", "anthropic": "Anthropic", "custom": "Custom"}

_FEATURE_LABELS = {
    "translate_draft": ("settings.route.draft", "Draft translation"),
    "translate_polish": ("settings.route.polish", "Polish pass"),
    "recap": ("settings.route.recap", "Recap scripting"),
    "poster": ("settings.route.poster", "Poster title design"),
}

_STATUS_COLORS = {
    "valid": "#3fb950",
    "invalid": "#f85149",
    "cooldown": "#d29922",
    "exhausted": "#8b949e",
    "unknown": "#8b949e",
}


def _mask_key(key: str) -> str:
    key = key or ""
    if len(key) <= 12:
        return key[:4] + "…" if len(key) > 4 else key
    return f"{key[:6]}…{key[-4:]}"


def _err_text(exc_or_en, km: str = "") -> str:
    if isinstance(exc_or_en, Exception):
        en = getattr(exc_or_en, "en", str(exc_or_en))
        km = getattr(exc_or_en, "km", "")
    else:
        en = str(exc_or_en)
    return km if (get_language() == "km" and km) else en


class _DownloadRow(QWidget):
    """One model-manager entry: name, size, installed flag, download w/ progress."""

    def __init__(self, tab: "SettingsTab", model_id: str, label: str,
                 size_mb: float | None, installed: bool):
        super().__init__()
        self._tab = tab
        self._id = model_id
        self._worker = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        size_txt = f"{size_mb:.0f} MB" if size_mb else "?"
        self.name_lbl = QLabel(f"{label}  ({size_txt})", self)
        lay.addWidget(self.name_lbl, 2)
        self.state_lbl = QLabel("", self)
        lay.addWidget(self.state_lbl, 1)
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 1000)
        self.bar.setVisible(False)
        lay.addWidget(self.bar, 2)
        self.btn = QPushButton("", self)
        self.btn.clicked.connect(self._download)
        lay.addWidget(self.btn)
        self.btn_cancel = QToolButton(self)
        self.btn_cancel.setText("X")
        self.btn_cancel.setToolTip(tr("settings.cancel", "Cancel"))
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel)
        lay.addWidget(self.btn_cancel)
        self.set_installed(installed)

    def set_installed(self, installed: bool) -> None:
        if installed:
            self.state_lbl.setText(tr("settings.model.installed", "Installed"))
            self.state_lbl.setStyleSheet("color: #3fb950;")
            self.btn.setText(tr("settings.model.redownload", "Re-download"))
        else:
            self.state_lbl.setText(tr("settings.model.missing", "Not installed"))
            self.state_lbl.setStyleSheet("color: #d29922;")
            self.btn.setText(tr("settings.model.download", "Download"))

    def _download(self) -> None:
        if self._worker is not None and getattr(self._worker, "isRunning", lambda: False)():
            return

        model_id = self._id

        def job(progress=None, cancel: threading.Event | None = None):
            from app.core.model_manager import download_model
            download_model(model_id, progress=progress, cancel=cancel)
            return True

        from app.ui.workers import FnWorker
        worker = FnWorker(job)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        self.bar.setVisible(True)
        self.bar.setValue(0)
        self.btn.setEnabled(False)
        self.btn_cancel.setVisible(True)
        self._tab._start_worker(worker)

    def _cancel(self) -> None:
        if self._worker is not None:
            try:
                self._worker.cancel()
            except Exception as exc:
                log.warning("model download cancel failed: %s", exc)

    def _on_progress(self, fraction: float, text: str) -> None:
        self.bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        if text:
            self.state_lbl.setText(text)
            self.state_lbl.setStyleSheet("")

    def _on_done(self, _result=None) -> None:
        self.bar.setVisible(False)
        self.btn.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.set_installed(True)

    def _on_failed(self, en: str, km: str) -> None:
        self.bar.setVisible(False)
        self.btn.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.state_lbl.setText(tr("settings.model.failed", "Failed"))
        self.state_lbl.setStyleSheet("color: #f85149;")
        QMessageBox.warning(self, tr("settings.model.download", "Download"),
                            km if (get_language() == "km" and km) else en)


class _ProviderCard(QGroupBox):
    """Card for a single AI provider: enable, base URL, keys, live checks."""

    def __init__(self, tab: "SettingsTab", provider: str):
        super().__init__(_PROVIDER_TITLES.get(provider, provider.capitalize()), tab)
        self._tab = tab
        self.provider = provider
        cfg = tab.ctx.config
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        self.chk_enabled = QCheckBox(tr("settings.provider.enabled", "Enabled"), self)
        self.chk_enabled.setChecked(bool(cfg.get(f"provider.{provider}.enabled", True)))
        self.chk_enabled.toggled.connect(
            lambda on: cfg.set(f"provider.{provider}.enabled", bool(on)))
        top.addWidget(self.chk_enabled)
        top.addWidget(QLabel(tr("settings.provider.base_url", "Base URL")))
        self.base_url = QLineEdit(self)
        self.base_url.setText(str(cfg.get(f"provider.{provider}.base_url",
                                          _DEFAULT_BASE_URLS.get(provider, "")) or ""))
        self.base_url.setPlaceholderText(_DEFAULT_BASE_URLS.get(provider, "https://..."))
        self.base_url.editingFinished.connect(
            lambda: cfg.set(f"provider.{provider}.base_url", self.base_url.text().strip()))
        top.addWidget(self.base_url, 1)
        lay.addLayout(top)

        if provider == "custom":
            hint_row = QHBoxLayout()
            hint_row.addWidget(QLabel(tr("settings.provider.model_hint", "Model hint")))
            self.model_hint = QLineEdit(self)
            self.model_hint.setText(str(cfg.get("provider.custom.model_hint", "") or ""))
            self.model_hint.setPlaceholderText("e.g. qwen2.5-72b-instruct")
            self.model_hint.editingFinished.connect(
                lambda: cfg.set("provider.custom.model_hint", self.model_hint.text().strip()))
            hint_row.addWidget(self.model_hint, 1)
            lay.addLayout(hint_row)

        self.keys_edit = QPlainTextEdit(self)
        self.keys_edit.setPlaceholderText(tr("settings.keys.ph", "One API key per line"))
        self.keys_edit.setMaximumHeight(84)
        lay.addWidget(self.keys_edit)

        btns = QHBoxLayout()
        b_save = QPushButton(tr("settings.save_keys", "Save keys"), self)
        b_save.clicked.connect(self.save_keys)
        btns.addWidget(b_save)
        b_import = QPushButton(tr("settings.import_keys", "Import Keys (.txt)"), self)
        b_import.clicked.connect(self.import_keys)
        btns.addWidget(b_import)
        self.b_check = QPushButton(tr("settings.check_keys", "Check Keys (Live)"), self)
        self.b_check.clicked.connect(self.check_keys)
        btns.addWidget(self.b_check)
        b_models = QPushButton(tr("settings.fetch_models", "Fetch Models"), self)
        b_models.clicked.connect(self.fetch_models)
        btns.addWidget(b_models)
        btns.addStretch(1)
        self.check_bar = QProgressBar(self)
        self.check_bar.setRange(0, 1000)
        self.check_bar.setVisible(False)
        self.check_bar.setMaximumWidth(180)
        btns.addWidget(self.check_bar)
        lay.addLayout(btns)

        self.keys_table = QTableWidget(0, 4, self)
        self.keys_table.setHorizontalHeaderLabels([
            tr("settings.keys.key", "Key"),
            tr("settings.keys.status", "Status"),
            tr("settings.keys.latency", "Latency"),
            tr("settings.keys.message", "Message"),
        ])
        kh = self.keys_table.horizontalHeader()
        kh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        kh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        kh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        kh.setSectionResizeMode(3, QHeaderView.Stretch)
        self.keys_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.keys_table.verticalHeader().setVisible(False)
        self.keys_table.setMaximumHeight(140)
        lay.addWidget(self.keys_table)

        self.reload_keys()

    # -- keys ------------------------------------------------------------
    def _current_keys(self) -> list:
        try:
            return self._tab.ctx.config.get_keys(self.provider)
        except Exception as exc:
            log.warning("get_keys(%s) failed: %s", self.provider, exc)
            return []

    def reload_keys(self) -> None:
        infos = self._current_keys()
        self.keys_edit.setPlainText("\n".join(k.key for k in infos))
        self.keys_table.setRowCount(0)
        exhausted = ""              # "credit" (no money left) / "quota" (resets) / ""
        for info in infos:
            row = self.keys_table.rowCount()
            self.keys_table.insertRow(row)
            self.keys_table.setItem(row, 0, QTableWidgetItem(_mask_key(info.key)))
            status = getattr(info, "status", "unknown") or "unknown"
            shown = status
            if status == "exhausted":
                if key_out_of_credit(info):
                    shown, exhausted = tr("settings.key.no_credit", "out of credit"), "credit"
                else:
                    shown = tr("settings.key.no_quota", "out of quota")
                    exhausted = exhausted or "quota"
            st = QTableWidgetItem(shown)
            st.setForeground(QColor(_STATUS_COLORS.get(status, "#8b949e")))
            self.keys_table.setItem(row, 1, st)
            lat = getattr(info, "latency_ms", None)
            self.keys_table.setItem(row, 2, QTableWidgetItem(f"{lat} ms" if lat else "-"))
            self.keys_table.setItem(row, 3, QTableWidgetItem(getattr(info, "last_error", "") or ""))
        self._tab.update_quota_banner(self.provider, exhausted)
        self._tab.refresh_default_provider_combo()

    def save_keys(self) -> None:
        lines: list[str] = []
        for ln in self.keys_edit.toPlainText().splitlines():
            ln = ln.strip()
            if ln and ln not in lines:
                lines.append(ln)
        cfg = self._tab.ctx.config
        try:
            existing = {k.key for k in self._current_keys()}
            wanted = set(lines)
            for key in existing - wanted:
                cfg.remove_key(self.provider, key)
            if lines:
                cfg.add_keys(self.provider, lines)
        except Exception as exc:
            QMessageBox.warning(self, tr("settings.save_keys", "Save keys"), _err_text(exc))
            return
        self.reload_keys()

    def import_keys(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("settings.import_keys", "Import Keys (.txt)"), "",
            "Text (*.txt);;All files (*.*)")
        if not path:
            return
        try:
            added = self._tab.ctx.config.import_keys_txt(self.provider, path)
        except Exception as exc:
            QMessageBox.warning(self, tr("settings.import_keys", "Import Keys (.txt)"),
                                _err_text(exc))
            return
        QMessageBox.information(
            self, tr("settings.import_keys", "Import Keys (.txt)"),
            tr("settings.keys.imported", "Imported {n} new key(s).").replace("{n}", str(added)))
        self.reload_keys()

    def check_keys(self) -> None:
        provider = self.provider
        ai = self._tab.ctx.ai

        def job(progress=None, cancel: threading.Event | None = None):
            return ai.check_keys_live(provider, progress=progress, cancel=cancel)

        from app.ui.workers import FnWorker
        worker = FnWorker(job)
        self.b_check.setEnabled(False)
        self.check_bar.setVisible(True)
        self.check_bar.setValue(0)
        worker.progress.connect(
            lambda f, t: self.check_bar.setValue(int(max(0.0, min(1.0, f)) * 1000)))

        def done(_results) -> None:
            self.b_check.setEnabled(True)
            self.check_bar.setVisible(False)
            self.reload_keys()

        def failed(en: str, km: str) -> None:
            self.b_check.setEnabled(True)
            self.check_bar.setVisible(False)
            QMessageBox.warning(self, tr("settings.check_keys", "Check Keys (Live)"),
                                km if (get_language() == "km" and km) else en)
            self.reload_keys()

        worker.finished_ok.connect(done)
        worker.failed.connect(failed)
        self._tab._start_worker(worker)

    def fetch_models(self) -> None:
        provider = self.provider
        ai = self._tab.ctx.ai

        def job(progress=None, cancel: threading.Event | None = None):
            return ai.list_models(provider)

        from app.ui.workers import FnWorker
        worker = FnWorker(job)

        def done(models) -> None:
            models = [str(m) for m in (models or [])]
            if not models:
                QMessageBox.information(
                    self, tr("settings.fetch_models", "Fetch Models"),
                    tr("settings.models.none", "No models returned."))
                return
            self._tab.ctx.config.set(f"provider.{provider}.models", models)
            self._tab.apply_models_to_routes(provider, models)

        worker.finished_ok.connect(done)
        worker.failed.connect(
            lambda en, km: QMessageBox.warning(
                self, tr("settings.fetch_models", "Fetch Models"),
                km if (get_language() == "km" and km) else en))
        self._tab._start_worker(worker)


class SettingsTab(QWidget):
    def __init__(self, ctx: "AppContext"):
        super().__init__()
        self.ctx = ctx
        self._workers: list = []
        self._exhausted: dict[str, str] = {}   # provider -> "credit" / "quota" / ""
        self._route_widgets: dict[str, dict] = {}
        self._loading_routes = False
        self._usage_baseline: dict[tuple[str, str], tuple[int, int, float]] = {}
        self._model_rows: dict[str, _DownloadRow] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # quota warning banner
        self.quota_banner = QLabel("", content)
        self.quota_banner.setWordWrap(True)
        self.quota_banner.setStyleSheet(
            "background:#3a2d12; color:#d29922; border:1px solid #d29922;"
            "border-radius:6px; padding:6px;")
        self.quota_banner.setVisible(False)
        root.addWidget(self.quota_banner)

        # (0) global default provider
        root.addWidget(self._section_label(
            tr("settings.default_ai", "Default AI")))
        root.addWidget(self._build_default_provider_card(content))

        # (1) provider cards
        root.addWidget(self._section_label(tr("settings.providers", "AI Providers")))
        self.cards: dict[str, _ProviderCard] = {}
        for p in PROVIDERS:
            card = _ProviderCard(self, p)
            self.cards[p] = card
            root.addWidget(card)

        # (2) routing grid
        root.addWidget(self._section_label(tr("settings.routing", "Per-feature routing")))
        root.addWidget(self._build_routing_grid(content))

        # (3) usage & cost
        root.addWidget(self._section_label(tr("settings.usage", "Usage & cost")))
        root.addWidget(self._build_usage_card(content))

        # (4) TTS / API extras
        root.addWidget(self._section_label(tr("settings.tts", "TTS / API extras")))
        root.addWidget(self._build_tts_card(content))

        # (5) app settings + model manager
        root.addWidget(self._section_label(tr("settings.app", "Application")))
        root.addWidget(self._build_app_card(content))
        root.addWidget(self._section_label(tr("settings.models", "Model manager")))
        root.addWidget(self._build_models_card(content))

        # (6) export / import settings
        root.addWidget(self._section_label(tr("settings.transfer", "Settings transfer")))
        root.addWidget(self._build_transfer_card(content))
        root.addStretch(1)

        self._load_routes()
        self._snapshot_usage_baseline()
        self.refresh_usage()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #58a6ff;"
                          "margin-top: 6px;")
        return lbl

    # ------------------------------------------------------------------
    # (0) global default provider
    # ------------------------------------------------------------------
    def _build_default_provider_card(self, parent: QWidget) -> QWidget:
        box = QGroupBox(tr("settings.default_ai.box", "Default AI provider"), parent)
        lay = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("settings.default_ai.label",
                                "AI used by every feature")))
        self.default_provider_combo = QComboBox(box)
        self.default_provider_combo.setMinimumWidth(240)
        row.addWidget(self.default_provider_combo, 1)
        self.b_apply_default = QPushButton(
            tr("settings.default_ai.apply", "Apply to all features"), box)
        self.b_apply_default.clicked.connect(self._apply_default_to_all)
        row.addWidget(self.b_apply_default)
        lay.addLayout(row)

        self.default_provider_hint = QLabel("", box)
        self.default_provider_hint.setWordWrap(True)
        self.default_provider_hint.setStyleSheet("color:#8b949e;")
        lay.addWidget(self.default_provider_hint)

        self.refresh_default_provider_combo()
        self.default_provider_combo.currentIndexChanged.connect(
            self._default_provider_changed)
        return box

    def retranslate(self) -> None:
        """The interface language changed (no restart): texts built from
        data are built again (app.ui.retranslate converted the rest)."""
        self.refresh_default_provider_combo()
        self._update_month_label()

    def refresh_default_provider_combo(self) -> None:
        """Rebuild the provider list, annotating how many keys each one holds."""
        combo = getattr(self, "default_provider_combo", None)
        if combo is None:
            return
        try:
            current = self.ctx.config.get("ai.default_provider") or None
        except Exception:
            current = None
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(tr("settings.default_ai.auto",
                         "Auto (first provider with keys)"), None)
        for p in PROVIDERS:
            try:
                n = len(self.ctx.config.get_keys(p))
            except Exception:
                n = 0
            if n:
                label = f"{p.capitalize()}  ({n} key{'s' if n != 1 else ''})"
            else:
                label = f"{p.capitalize()}  " + tr("settings.default_ai.no_keys",
                                                   "(no keys)")
            combo.addItem(label, p)
            if not n:
                combo.setItemData(combo.count() - 1, QColor("#8b949e"),
                                  Qt.ForegroundRole)
        idx = combo.findData(current) if current else 0
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
        self._update_default_provider_hint()

    def _default_provider(self) -> str:
        """Provider the app would pick for a feature with no explicit routing."""
        try:
            return self.ctx.config._default_route().provider
        except Exception as exc:
            log.warning("default provider probe failed: %s", exc)
            return PROVIDERS[0]

    def _update_default_provider_hint(self) -> None:
        lbl = getattr(self, "default_provider_hint", None)
        if lbl is None:
            return
        try:
            route = self.ctx.config._default_route()
            has_keys = bool(self.ctx.config.get_keys(route.provider))
        except Exception as exc:
            log.warning("default route probe failed: %s", exc)
            lbl.setText("")
            return
        if has_keys:
            en = ("Features without their own routing will use "
                  f"{route.provider} / {route.model}.")
            km = ("មុខងារដែលមិនបានកំណត់ដោយខ្លួនឯង នឹងប្រើ "
                  f"{route.provider} / {route.model}។")
            lbl.setStyleSheet("color:#8b949e;")
        else:
            en = (f"{route.provider} has no API keys yet. Add keys below, "
                  "or choose a provider that already has them.")
            km = (f"{route.provider} មិនទាន់មាន API key ទេ។ "
                  "សូមបញ្ចូល key ខាងក្រោម ឬជ្រើសរើសក្រុមហ៊ុនផ្សេង។")
            lbl.setStyleSheet("color:#d29922;")
        lbl.setText(km if get_language() == "km" else en)

    def _default_provider_changed(self, _idx: int) -> None:
        prov = self.default_provider_combo.currentData()
        try:
            self.ctx.config.set("ai.default_provider", prov or "")
        except Exception as exc:
            log.warning("set ai.default_provider failed: %s", exc)
        self._update_default_provider_hint()
        self._load_routes()

    def _apply_default_to_all(self) -> None:
        title = tr("settings.default_ai.apply", "Apply to all features")
        en = ("Clear per-feature routing so every feature follows the default AI?\n"
              "Existing per-feature choices will be lost.")
        km = ("លុបការកំណត់ AI តាមមុខងារ ដើម្បីឱ្យមុខងារទាំងអស់ប្រើ AI លំនាំដើម?\n"
              "ការជ្រើសរើសតាមមុខងារនឹងត្រូវបាត់បង់។")
        msg = km if get_language() == "km" else en
        if QMessageBox.question(self, title, msg) != QMessageBox.Yes:
            return
        try:
            self.ctx.config.clear_routes()
        except Exception as exc:
            QMessageBox.warning(self, title, _err_text(exc))
            return
        self._load_routes()
        self._update_default_provider_hint()

    def _start_worker(self, worker) -> None:
        started = False
        for meth in ("start",):
            fn = getattr(self.ctx.pool, meth, None)
            if callable(fn):
                try:
                    fn(worker)
                    started = True
                    break
                except TypeError:
                    break
                except Exception:
                    break
        if not started:
            worker.start()
        for meth in ("add", "keep", "track"):
            fn = getattr(self.ctx.pool, meth, None)
            if callable(fn):
                try:
                    fn(worker)
                    break
                except Exception:
                    continue
        self._workers.append(worker)
        self._workers = [w for w in self._workers
                         if getattr(w, "isRunning", lambda: True)() or w is worker]

    def update_quota_banner(self, provider: str, exhausted: str) -> None:
        """`exhausted`: "credit" (the account has no money left), "quota"
        (a limit that resets by itself) or "" for `provider`'s keys."""
        self._exhausted[provider] = exhausted

        def names(kind: str) -> str:
            return ", ".join(PROVIDER_NAMES.get(p, p)
                             for p, e in sorted(self._exhausted.items()) if e == kind)

        lines = []
        if names("credit"):
            lines.append(tr("settings.no_credit_warning",
                            "Out of credit: {names}. Add credit on the provider's website, "
                            "or add a key of another provider.")
                         .replace("{names}", names("credit")))
        if names("quota"):
            lines.append(tr("settings.quota_warning",
                            "Quota used up for now: {names}. It resets by itself, or add a "
                            "key of another provider.")
                         .replace("{names}", names("quota")))
        self.quota_banner.setText("\n".join(lines))
        self.quota_banner.setVisible(bool(lines))

    # ------------------------------------------------------------------
    # (2) routing
    # ------------------------------------------------------------------
    def _build_routing_grid(self, parent: QWidget) -> QWidget:
        box = QGroupBox(tr("settings.routing.box", "Feature routing"), parent)
        grid = QGridLayout(box)
        grid.addWidget(QLabel(tr("settings.route.feature", "Feature")), 0, 0)
        grid.addWidget(QLabel(tr("settings.route.provider", "Provider")), 0, 1)
        grid.addWidget(QLabel(tr("settings.route.model", "Model")), 0, 2)
        grid.addWidget(QLabel(tr("settings.route.fb_provider", "Fallback provider")), 0, 3)
        grid.addWidget(QLabel(tr("settings.route.fb_model", "Fallback model")), 0, 4)

        for i, feature in enumerate(FEATURES, start=1):
            key, default = _FEATURE_LABELS.get(feature, ("", feature))
            grid.addWidget(QLabel(tr(key, default)), i, 0)

            prov = QComboBox(box)
            prov.addItems(PROVIDERS)
            grid.addWidget(prov, i, 1)

            model = QComboBox(box)
            model.setEditable(True)
            grid.addWidget(model, i, 2)

            fb_prov = QComboBox(box)
            fb_prov.addItem(tr("settings.route.none", "None"), None)
            for p in PROVIDERS:
                fb_prov.addItem(p, p)
            grid.addWidget(fb_prov, i, 3)

            fb_model = QComboBox(box)
            fb_model.setEditable(True)
            grid.addWidget(fb_model, i, 4)

            self._route_widgets[feature] = {
                "provider": prov, "model": model,
                "fb_provider": fb_prov, "fb_model": fb_model,
            }
            prov.currentIndexChanged.connect(
                lambda _i, f=feature: self._route_provider_changed(f, primary=True))
            fb_prov.currentIndexChanged.connect(
                lambda _i, f=feature: self._route_provider_changed(f, primary=False))
            model.currentTextChanged.connect(lambda _t, f=feature: self._route_changed(f))
            fb_model.currentTextChanged.connect(lambda _t, f=feature: self._route_changed(f))
        return box

    def _cached_models(self, provider: str) -> list[str]:
        try:
            models = self.ctx.config.get(f"provider.{provider}.models") or []
            return [str(m) for m in models]
        except Exception:
            return []

    def _fill_model_combo(self, combo: QComboBox, provider: str | None,
                          keep_text: bool = True) -> None:
        text = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if provider:
            combo.addItems(self._cached_models(provider))
        if keep_text and text:
            combo.setCurrentText(text)
        combo.blockSignals(False)

    def _route_provider_changed(self, feature: str, primary: bool) -> None:
        w = self._route_widgets[feature]
        if primary:
            self._fill_model_combo(w["model"], w["provider"].currentText(), keep_text=False)
        else:
            self._fill_model_combo(w["fb_model"], w["fb_provider"].currentData(), keep_text=False)
        self._route_changed(feature)

    def _route_changed(self, feature: str) -> None:
        if self._loading_routes:
            return
        w = self._route_widgets[feature]
        fb_provider = w["fb_provider"].currentData()
        fb_model = w["fb_model"].currentText().strip() or None
        route = Route(
            provider=w["provider"].currentText(),
            model=w["model"].currentText().strip(),
            fallback_provider=fb_provider,
            fallback_model=fb_model if fb_provider else None,
        )
        try:
            self.ctx.config.set_route(feature, route)
        except Exception as exc:
            log.warning("set_route(%s) failed: %s", feature, exc)

    def _load_routes(self) -> None:
        self._loading_routes = True
        try:
            for feature, w in self._route_widgets.items():
                try:
                    route = self.ctx.config.get_route(feature)
                except Exception as exc:
                    log.warning("get_route(%s) failed: %s", feature, exc)
                    continue
                if route is None:
                    continue
                provider = getattr(route, "provider", "") or self._default_provider()
                idx = w["provider"].findText(provider)
                if idx >= 0:
                    w["provider"].setCurrentIndex(idx)
                self._fill_model_combo(w["model"], provider, keep_text=False)
                w["model"].setCurrentText(getattr(route, "model", "") or "")
                fbp = getattr(route, "fallback_provider", None)
                fidx = w["fb_provider"].findData(fbp)
                w["fb_provider"].setCurrentIndex(fidx if fidx >= 0 else 0)
                self._fill_model_combo(w["fb_model"], fbp, keep_text=False)
                w["fb_model"].setCurrentText(getattr(route, "fallback_model", "") or "")
        finally:
            self._loading_routes = False

    def apply_models_to_routes(self, provider: str, models: list[str]) -> None:
        """Called after Fetch Models: refresh combos where this provider is chosen."""
        for w in self._route_widgets.values():
            if w["provider"].currentText() == provider:
                self._fill_model_combo(w["model"], provider, keep_text=True)
            if w["fb_provider"].currentData() == provider:
                self._fill_model_combo(w["fb_model"], provider, keep_text=True)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._load_routes()
        self.refresh_usage()

    # ------------------------------------------------------------------
    # (3) usage & cost
    # ------------------------------------------------------------------
    def _build_usage_card(self, parent: QWidget) -> QWidget:
        box = QGroupBox(tr("settings.usage.box", "Token usage"), parent)
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        b_refresh = QPushButton(tr("settings.usage.refresh", "Refresh"), box)
        b_refresh.clicked.connect(self.refresh_usage)
        row.addWidget(b_refresh)
        b_reset = QPushButton(tr("settings.usage.reset", "Reset session"), box)
        b_reset.clicked.connect(self._reset_session)
        row.addWidget(b_reset)
        row.addStretch(1)
        lay.addLayout(row)

        self.usage_table = QTableWidget(0, 5, box)
        self.usage_table.setHorizontalHeaderLabels([
            tr("settings.usage.provider", "Provider"),
            tr("settings.usage.model", "Model"),
            tr("settings.usage.in", "In tokens"),
            tr("settings.usage.out", "Out tokens"),
            tr("settings.usage.cost", "Est. cost"),
        ])
        uh = self.usage_table.horizontalHeader()
        uh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        uh.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in (2, 3, 4):
            uh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.usage_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.usage_table.verticalHeader().setVisible(False)
        self.usage_table.setMaximumHeight(220)
        lay.addWidget(self.usage_table)

        brow = QHBoxLayout()
        brow.addWidget(QLabel(tr("settings.budget.limit", "Monthly AI budget:"), box))
        self.budget_spin = QDoubleSpinBox(box)
        self.budget_spin.setRange(0.0, 100000.0)
        self.budget_spin.setDecimals(2)
        self.budget_spin.setPrefix("$")
        self.budget_spin.setSpecialValueText(tr("settings.budget.none", "No limit"))
        limit, warn = self.ctx.config.get_budget()
        self.budget_spin.setValue(limit)
        self.budget_spin.setToolTip(tr(
            "settings.budget.tip",
            "AI calls stop once this month's estimated cost reaches the limit. "
            "A warning appears in the console at the warning level."))
        brow.addWidget(self.budget_spin)
        brow.addWidget(QLabel(tr("settings.budget.warn", "Warn at:"), box))
        self.budget_warn = QDoubleSpinBox(box)
        self.budget_warn.setRange(10, 100)
        self.budget_warn.setDecimals(0)
        self.budget_warn.setSuffix(" %")
        self.budget_warn.setValue(round(warn * 100))
        brow.addWidget(self.budget_warn)
        self.month_label = QLabel("", box)
        self.month_label.setStyleSheet("color: #8b949e;")
        brow.addWidget(self.month_label, 1)
        self.budget_spin.valueChanged.connect(self._save_budget)
        self.budget_warn.valueChanged.connect(self._save_budget)
        lay.addLayout(brow)
        self._update_month_label()
        return box

    def _save_budget(self, *_a) -> None:
        self.ctx.config.set("budget.monthly_usd", float(self.budget_spin.value()))
        self.ctx.config.set("budget.warn_fraction", float(self.budget_warn.value()) / 100.0)
        self._update_month_label()

    def _update_month_label(self) -> None:
        try:
            spent = self.ctx.config.month_cost()
        except Exception:
            return
        limit, _w = self.ctx.config.get_budget()
        text = tr("settings.budget.month", "This month") + f": ${spent:,.2f}"
        if limit > 0:
            text += f" / ${limit:,.2f} ({spent / limit:.0%})"
            color = "#f85149" if spent >= limit else (
                "#d29922" if spent >= limit * self.budget_warn.value() / 100 else "#8b949e")
            self.month_label.setStyleSheet(f"color: {color};")
        try:
            from app.core.translation_memory import get_tm
            st = get_tm().stats()
            text += ("   ·   " + tr("settings.tm.stats", "Translation memory")
                     + f": {st['entries']:,} ({st['human']:,} "
                     + tr("settings.tm.human", "your fixes") + f", {st['reused']:,} "
                     + tr("settings.tm.reused", "reused") + ")")
        except Exception:
            pass
        self.month_label.setText(text)

    def _usage_rows(self) -> list[dict]:
        try:
            return self.ctx.config.get_usage() or []
        except Exception as exc:
            log.warning("get_usage failed: %s", exc)
            return []

    def _snapshot_usage_baseline(self) -> None:
        self._usage_baseline = {}
        for r in self._usage_rows():
            key = (str(r.get("provider", "")), str(r.get("model", "")))
            self._usage_baseline[key] = (
                int(r.get("in_tokens", 0) or 0),
                int(r.get("out_tokens", 0) or 0),
                float(r.get("est_cost", 0.0) or 0.0),
            )

    def _reset_session(self) -> None:
        self._snapshot_usage_baseline()
        self.refresh_usage()

    def refresh_usage(self) -> None:
        rows = self._usage_rows()
        self.usage_table.setRowCount(0)
        sess_in = sess_out = 0
        sess_cost = 0.0
        for r in rows:
            row = self.usage_table.rowCount()
            self.usage_table.insertRow(row)
            self.usage_table.setItem(row, 0, QTableWidgetItem(str(r.get("provider", ""))))
            self.usage_table.setItem(row, 1, QTableWidgetItem(str(r.get("model", ""))))
            in_t = int(r.get("in_tokens", 0) or 0)
            out_t = int(r.get("out_tokens", 0) or 0)
            cost = float(r.get("est_cost", 0.0) or 0.0)
            self.usage_table.setItem(row, 2, QTableWidgetItem(f"{in_t:,}"))
            self.usage_table.setItem(row, 3, QTableWidgetItem(f"{out_t:,}"))
            self.usage_table.setItem(row, 4, QTableWidgetItem(f"${cost:,.4f}"))
            base = self._usage_baseline.get(
                (str(r.get("provider", "")), str(r.get("model", ""))), (0, 0, 0.0))
            sess_in += max(0, in_t - base[0])
            sess_out += max(0, out_t - base[1])
            sess_cost += max(0.0, cost - base[2])
        # session total row
        row = self.usage_table.rowCount()
        self.usage_table.insertRow(row)
        total_item = QTableWidgetItem(tr("settings.usage.session", "Session total"))
        total_item.setForeground(QColor("#58a6ff"))
        self.usage_table.setItem(row, 0, total_item)
        self.usage_table.setItem(row, 1, QTableWidgetItem(""))
        self.usage_table.setItem(row, 2, QTableWidgetItem(f"{sess_in:,}"))
        self.usage_table.setItem(row, 3, QTableWidgetItem(f"{sess_out:,}"))
        self.usage_table.setItem(row, 4, QTableWidgetItem(f"${sess_cost:,.4f}"))
        for c in range(2, 5):
            it = self.usage_table.item(row, c)
            if it:
                it.setForeground(QColor("#58a6ff"))
        if hasattr(self, "month_label"):
            self._update_month_label()

    # ------------------------------------------------------------------
    # (4) TTS extras
    # ------------------------------------------------------------------
    def _build_tts_card(self, parent: QWidget) -> QWidget:
        cfg = self.ctx.config
        box = QGroupBox(tr("settings.speech.box", "Speech services"), parent)
        grid = QGridLayout(box)

        grid.addWidget(QLabel(tr("settings.stt", "Speech-to-text")), 0, 0)
        self.stt_combo = QComboBox(box)
        self.stt_combo.addItem(tr("settings.stt.whisper", "Whisper — on this PC, free"), "whisper")
        self.stt_combo.addItem(tr("settings.stt.deepgram",
                                  "Deepgram — online, fast, also tells who speaks"), "deepgram")
        i = self.stt_combo.findData(str(cfg.get("stt.engine") or "whisper"))
        self.stt_combo.setCurrentIndex(max(0, i))
        self.stt_combo.setToolTip(tr(
            "settings.stt.tip", "Used by Full Dub, Recap and transcription. Deepgram "
            "hears Chinese, Thai, Vietnamese, Korean, Japanese, English and more, but not "
            "Khmer (Whisper is used then). It is billed per audio minute."))
        self.stt_combo.currentIndexChanged.connect(
            lambda _i: cfg.set("stt.engine", self.stt_combo.currentData()))
        grid.addWidget(self.stt_combo, 0, 1, 1, 3)

        grid.addWidget(QLabel(tr("settings.deepgram", "Deepgram API key")), 1, 0)
        self.deepgram_edit = QLineEdit(box)
        self.deepgram_edit.setEchoMode(QLineEdit.Password)
        self.deepgram_edit.setText(cfg.get_secret("deepgram.api_key"))
        grid.addWidget(self.deepgram_edit, 1, 1)
        b = QPushButton(tr("settings.tts.save", "Save"), box)
        b.clicked.connect(lambda: self._save_secret("deepgram.api_key", self.deepgram_edit))
        grid.addWidget(b, 1, 2)
        self.b_check_dg = QPushButton(tr("settings.check", "Check"), box)
        self.b_check_dg.clicked.connect(self._check_deepgram)
        grid.addWidget(self.b_check_dg, 1, 3)

        grid.addWidget(QLabel(tr("settings.tts.eleven", "ElevenLabs API key")), 2, 0)
        self.eleven_edit = QLineEdit(box)
        self.eleven_edit.setEchoMode(QLineEdit.Password)
        self.eleven_edit.setText(cfg.get_secret("elevenlabs.api_key"))
        grid.addWidget(self.eleven_edit, 2, 1)
        b_save = QPushButton(tr("settings.tts.save", "Save"), box)
        b_save.clicked.connect(lambda: self._save_secret("elevenlabs.api_key", self.eleven_edit))
        grid.addWidget(b_save, 2, 2)
        # a key saved by an older version is plain text: store it encrypted now
        if self.eleven_edit.text() and not str(cfg.get("elevenlabs.api_key") or "").startswith(
                ("dpapi:", "b64:")):
            cfg.set_secret("elevenlabs.api_key", self.eleven_edit.text())

        self.speech_status = QLabel("", box)
        self.speech_status.setWordWrap(True)
        grid.addWidget(self.speech_status, 3, 0, 1, 4)
        note = QLabel(tr("settings.tts.openai_note",
                         "OpenAI TTS uses the OpenAI key pool configured above."), box)
        note.setStyleSheet("color: #8b949e;")
        grid.addWidget(note, 4, 0, 1, 4)
        return box

    def _save_secret(self, key: str, edit: QLineEdit) -> None:
        self.ctx.config.set_secret(key, edit.text())
        self.speech_status.setStyleSheet("color: #3fb950;")
        self.speech_status.setText(tr("settings.saved_enc", "Saved (encrypted on this PC)."))

    def _check_deepgram(self) -> None:
        key = self.deepgram_edit.text().strip()
        self.b_check_dg.setEnabled(False)
        self.speech_status.setStyleSheet("color: #8b949e;")
        self.speech_status.setText(tr("settings.checking", "Checking…"))

        def job():
            from app.engines.deepgram_stt import check_key
            return check_key(key)

        def ok(msg) -> None:
            self.b_check_dg.setEnabled(True)
            self.speech_status.setStyleSheet("color: #3fb950;")
            self.speech_status.setText(str(msg))

        def bad(en: str, km: str) -> None:
            self.b_check_dg.setEnabled(True)
            self.speech_status.setStyleSheet("color: #f85149;")
            self.speech_status.setText(km if i18n.get_language() == "km" and km else en)

        from app.ui.workers import FnWorker
        w = FnWorker(job)
        w.finished_ok.connect(ok)
        w.failed.connect(bad)
        self.ctx.pool.start(w)

    # ------------------------------------------------------------------
    # (5) app settings + model manager
    # ------------------------------------------------------------------
    def _build_app_card(self, parent: QWidget) -> QWidget:
        box = QGroupBox(tr("settings.app.box", "Application settings"), parent)
        grid = QGridLayout(box)
        cfg = self.ctx.config

        grid.addWidget(QLabel(tr("settings.language", "UI language")), 0, 0)
        self.lang_combo = QComboBox(box)
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("ខ្មែរ", "km")
        cur = i18n.get_language()
        self.lang_combo.setCurrentIndex(1 if cur == "km" else 0)
        self.lang_combo.currentIndexChanged.connect(self._language_changed)
        grid.addWidget(self.lang_combo, 0, 1)

        self.chk_hw = QCheckBox(tr("settings.hw", "Prefer hardware encoding (NVENC/QSV)"), box)
        self.chk_hw.setChecked(bool(cfg.get("export.hw", True)))
        self.chk_hw.toggled.connect(lambda on: cfg.set("export.hw", bool(on)))
        grid.addWidget(self.chk_hw, 1, 0, 1, 2)

        grid.addWidget(QLabel(tr("settings.duck", "Default ducking (dB)")), 2, 0)
        self.duck_spin = QDoubleSpinBox(box)
        self.duck_spin.setRange(-60.0, 0.0)
        self.duck_spin.setSingleStep(1.0)
        self.duck_spin.setValue(float(cfg.get("duck.db", -18.0) or -18.0))
        self.duck_spin.valueChanged.connect(lambda v: cfg.set("duck.db", float(v)))
        grid.addWidget(self.duck_spin, 2, 1)

        grid.addWidget(QLabel(tr("settings.dl_dir", "Downloads folder")), 3, 0)
        dl_row = QHBoxLayout()
        self.dl_dir_edit = QLineEdit(box)
        self.dl_dir_edit.setText(str(cfg.get("downloads.dir", "") or ""))
        self.dl_dir_edit.editingFinished.connect(
            lambda: self._set_dl_dir(self.dl_dir_edit.text().strip()))
        dl_row.addWidget(self.dl_dir_edit, 1)
        b_pick = QPushButton(tr("settings.browse", "Browse..."), box)
        b_pick.clicked.connect(self._pick_dl_dir)
        dl_row.addWidget(b_pick)
        grid.addLayout(dl_row, 3, 1)

        misc = QHBoxLayout()
        b_models_dir = QPushButton(tr("settings.open_models", "Open models folder"), box)
        b_models_dir.clicked.connect(self._open_models_dir)
        misc.addWidget(b_models_dir)
        b_env = QPushButton(tr("settings.check_env", "Check environment"), box)
        b_env.clicked.connect(self._check_environment)
        misc.addWidget(b_env)
        misc.addStretch(1)
        grid.addLayout(misc, 4, 0, 1, 2)
        return box

    def _language_changed(self, _idx: int) -> None:
        code = self.lang_combo.currentData() or "en"
        try:
            i18n.set_language(code)
            i18n.reload_catalogs()
        except Exception as exc:
            log.warning("set_language failed: %s", exc)
        en = "Language saved. Please restart the app to apply the new language."
        km = "បានរក្សាទុកភាសា។ សូមចាប់ផ្តើមកម្មវិធីឡើងវិញ ដើម្បីអនុវត្តភាសាថ្មី។"
        QMessageBox.information(self, tr("settings.language", "UI language"),
                                km if code == "km" else tr("settings.restart", en))

    def _pick_dl_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, tr("settings.dl_dir", "Downloads folder"),
            self.dl_dir_edit.text().strip() or os.path.expanduser("~"))
        if d:
            self.dl_dir_edit.setText(d)
            self._set_dl_dir(d)

    def _set_dl_dir(self, d: str) -> None:
        """Save the folder and point the running download queue at it."""
        self.ctx.config.set("downloads.dir", d)
        try:
            self.ctx.downloads.set_out_dir(d or None)
        except Exception as exc:
            log.warning("could not use downloads folder %r: %s", d, exc)
            QMessageBox.warning(self, tr("settings.dl_dir", "Downloads folder"), str(exc))

    def _open_models_dir(self) -> None:
        try:
            os.startfile(str(models_dir()))  # noqa: S606
        except Exception as exc:
            log.warning("open models dir failed: %s", exc)

    def _check_environment(self) -> None:
        def job(progress=None, cancel: threading.Event | None = None):
            from app.core.model_manager import startup_report
            return startup_report()

        from app.ui.workers import FnWorker
        worker = FnWorker(job)

        def done(report) -> None:
            report = report or {}
            gpu = report.get("gpu") or tr("settings.env.no_gpu", "None detected")
            vram = report.get("vram_gb")
            cuda = report.get("cuda")
            disk = report.get("disk_free_gb")
            missing = report.get("missing_models") or []
            ffmpeg_ok = report.get("ffmpeg_ok")
            acc = report.get("accel") or {}
            others = [g.get("name", "") for g in report.get("gpus") or []
                      if g.get("name") != report.get("gpu")]
            lines = [
                f"GPU: {gpu}" + (f" ({vram:.1f} GB VRAM)" if isinstance(vram, (int, float)) and vram else ""),
            ]
            if others:
                lines.append(tr("settings.env.other_gpus", "Other GPUs: ") + ", ".join(others))
            lines.append(f"CUDA: {'yes' if cuda else 'no'}")
            if acc:
                lines += [
                    tr("settings.env.whisper", "Speech recognition: ") + acc.get("whisper", "CPU"),
                    tr("settings.env.torch", "Voice separation / cloning: ") + acc.get("torch", "CPU"),
                    tr("settings.env.onnx", "Poster inpainting: ") + acc.get("onnx", "CPU"),
                    tr("settings.env.encoders", "GPU video encoders: ")
                    + (", ".join(acc.get("encoders") or []) or tr("settings.env.none", "none")),
                ]
            lines += [f"→ {h}" for h in report.get("accel_hints") or []]
            lines += [
                f"Disk free: {disk:.1f} GB" if isinstance(disk, (int, float)) else "Disk free: ?",
                f"FFmpeg: {'OK' if ffmpeg_ok else 'MISSING'}",
                (tr("settings.env.missing", "Missing models: ") + ", ".join(missing))
                if missing else tr("settings.env.all_models", "All models installed."),
            ]
            QMessageBox.information(self, tr("settings.check_env", "Check environment"),
                                    "\n".join(lines))

        worker.finished_ok.connect(done)
        worker.failed.connect(
            lambda en, km: QMessageBox.warning(
                self, tr("settings.check_env", "Check environment"),
                km if (get_language() == "km" and km) else en))
        self._start_worker(worker)

    def _build_models_card(self, parent: QWidget) -> QWidget:
        box = QGroupBox(tr("settings.models.box", "AI model downloads"), parent)
        lay = QVBoxLayout(box)
        try:
            from app.core.model_manager import MODELS
            entries = dict(MODELS)
        except Exception as exc:
            log.warning("model_manager MODELS unavailable: %s", exc)
            entries = {}
        if not entries:
            lay.addWidget(QLabel(tr("settings.models.empty",
                                    "Model manager is not available.")))
            return box
        for model_id, meta in entries.items():
            meta = meta or {}
            label = str(meta.get("label", model_id))
            size_mb = meta.get("size_mb")
            installed = False
            check = meta.get("check")
            if callable(check):
                try:
                    installed = bool(check())
                except Exception:
                    installed = False
            row = _DownloadRow(self, model_id, label,
                               float(size_mb) if size_mb else None, installed)
            self._model_rows[model_id] = row
            lay.addWidget(row)
        return box

    # ------------------------------------------------------------------
    # (6) export / import settings
    # ------------------------------------------------------------------
    def _build_transfer_card(self, parent: QWidget) -> QWidget:
        box = QGroupBox(tr("settings.transfer.box", "Backup / restore settings"), parent)
        row = QHBoxLayout(box)
        b_export = QPushButton(tr("settings.export", "Export Settings"), box)
        b_export.clicked.connect(self._export_settings)
        row.addWidget(b_export)
        b_import = QPushButton(tr("settings.import", "Import Settings"), box)
        b_import.clicked.connect(self._import_settings)
        row.addWidget(b_import)
        row.addStretch(1)
        return box

    def _export_settings(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("settings.export", "Export Settings"))
        lay = QVBoxLayout(dlg)
        chk = QCheckBox(tr("settings.export.include_keys",
                           "Include API keys (plaintext) / បញ្ចូល API keys"), dlg)
        lay.addWidget(chk)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        if dlg.exec() != QDialog.Accepted:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("settings.export", "Export Settings"),
            "recapstudio_settings.json", "JSON (*.json);;All files (*.*)")
        if not path:
            return
        try:
            self.ctx.config.export_settings(path, include_keys=chk.isChecked())
        except Exception as exc:
            QMessageBox.warning(self, tr("settings.export", "Export Settings"),
                                _err_text(exc))
            return
        QMessageBox.information(self, tr("settings.export", "Export Settings"),
                                tr("settings.export.done", "Settings exported."))

    def _import_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("settings.import", "Import Settings"), "",
            "JSON (*.json);;All files (*.*)")
        if not path:
            return
        try:
            self.ctx.config.import_settings(path)
        except Exception as exc:
            QMessageBox.warning(self, tr("settings.import", "Import Settings"),
                                _err_text(exc))
            return
        en = "Settings imported. Please restart the app to apply them."
        km = "បាននាំចូលការកំណត់។ សូមចាប់ផ្តើមកម្មវិធីឡើងវិញ ដើម្បីអនុវត្ត។"
        QMessageBox.information(self, tr("settings.import", "Import Settings"),
                                km if get_language() == "km" else
                                tr("settings.import.restart", en))
        for card in self.cards.values():
            card.reload_keys()
        self._load_routes()
        self.refresh_usage()
