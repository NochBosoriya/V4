"""Switch the interface language live: every text a widget shows is
converted with i18n.convert() (labels, buttons, group titles, tab names,
tooltips, placeholders, drop-down items, table / tree headers, dock and
window titles, actions). User data is left alone: line-edit and text-box
contents, table cells, tree and list items (tabs that build those from the
language re-fill them in their retranslate() hook), and drop-down items
that are their own value ("auto", "large-v3").

Each widget remembers its texts per language (a dynamic property), so
switching back gives exactly the text it had: Khmer uses one word where
English has two ("សំឡេង៖" = "Voice:" and "Voices:").
"""
from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QComboBox,
    QDockWidget,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QWidget,
)

from app.core.i18n import convert, is_ui_text

_MEM = "i18nTexts"          # dynamic property: {slot: {language: text}}


def retranslate_tree(root: QWidget, src: str, dst: str) -> int:
    """Convert every text under `root` from language `src` to `dst`.
    Returns how many texts changed."""
    changed = 0

    def conv(obj, slot: str, text: str) -> str:
        """`text` (shown in `src`) in `dst`, as this object had it before
        when it was shown in `dst`, else converted."""
        nonlocal changed
        if not text:
            return text
        mem = obj.property(_MEM)
        mem = dict(mem) if isinstance(mem, dict) else {}
        known = dict(mem.get(slot) or {})
        if known.get(src) == text and dst in known:
            new = known[dst]
        else:
            new = convert(text, src, dst)
            known = {src: text}
        known[dst] = new
        mem[slot] = known
        obj.setProperty(_MEM, mem)
        if new != text:
            changed += 1
        return new

    for w in [root, *root.findChildren(QWidget)]:
        try:
            tip = w.toolTip()
            if tip:
                w.setToolTip(conv(w, "tip", tip))
            if w.isWindow() and w.windowTitle():
                w.setWindowTitle(conv(w, "window", w.windowTitle()))
            if isinstance(w, QLabel):
                if w.text():
                    w.setText(conv(w, "text", w.text()))
            elif isinstance(w, QAbstractButton):
                if w.text():
                    w.setText(conv(w, "text", w.text()))
            elif isinstance(w, QGroupBox):
                if w.title():
                    w.setTitle(conv(w, "title", w.title()))
            elif isinstance(w, (QLineEdit, QPlainTextEdit, QTextEdit)):
                if w.placeholderText():
                    w.setPlaceholderText(conv(w, "ph", w.placeholderText()))
            elif isinstance(w, QComboBox):
                if not w.isEditable():          # editable = names the user typed
                    for i in range(w.count()):
                        text = w.itemText(i)
                        # an item without data may be its own value ("auto",
                        # "large-v3", read back with currentText()): only a
                        # text tr() made ("— select —") is converted then
                        if w.itemData(i) is None and not is_ui_text(text):
                            continue
                        w.setItemText(i, conv(w, f"item{i}", text))
                        t = w.itemData(i, 3)    # Qt.ToolTipRole
                        if isinstance(t, str) and t:
                            w.setItemData(i, conv(w, f"itemtip{i}", t), 3)
                elif w.lineEdit() is not None and w.lineEdit().placeholderText():
                    le = w.lineEdit()
                    le.setPlaceholderText(conv(le, "ph", le.placeholderText()))
            elif isinstance(w, QAbstractSpinBox):
                if w.specialValueText():
                    w.setSpecialValueText(conv(w, "special", w.specialValueText()))
            elif isinstance(w, QTabWidget):
                for i in range(w.count()):
                    w.setTabText(i, conv(w, f"tab{i}", w.tabText(i)))
                    if w.tabToolTip(i):
                        w.setTabToolTip(i, conv(w, f"tabtip{i}", w.tabToolTip(i)))
            elif isinstance(w, QDockWidget):
                w.setWindowTitle(conv(w, "dock", w.windowTitle()))
            if isinstance(w, QTableWidget):
                for i in range(w.columnCount()):
                    it = w.horizontalHeaderItem(i)
                    if it is not None:
                        it.setText(conv(w, f"head{i}", it.text()))
                        if it.toolTip():
                            it.setToolTip(conv(w, f"headtip{i}", it.toolTip()))
            elif isinstance(w, QTreeWidget):
                head = w.headerItem()
                if head is not None:
                    for i in range(head.columnCount()):
                        head.setText(i, conv(w, f"head{i}", head.text(i)))
        except RuntimeError:          # a widget deleted meanwhile
            continue
    for a in root.findChildren(QAction):
        try:
            if a.text():
                a.setText(conv(a, "text", a.text()))
            if a.toolTip() and a.toolTip() != a.text():
                a.setToolTip(conv(a, "tip", a.toolTip()))
        except RuntimeError:
            continue
    return changed
