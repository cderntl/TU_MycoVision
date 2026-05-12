# ==================== TU_MyCo-Vision v2 (PyQt5) ====================
#
# Requirements:
# Python 3.10 - 3.12 recommended
# pip install PyQt5 Pillow opencv-python plotly pandas numpy scipy ultralytics
#
# Written by:
# Kartik Deopujari
# Matthias Schmal
# Good coding vibes
# PyQt5 refactor: automated refactoring pass
# ====================================================================

import sys
import os
import time
import csv
import threading
import tempfile
import webbrowser
from pathlib import Path

# ── PyQt5 imports ──────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QLineEdit, QSlider,
    QTabWidget, QFileDialog, QMessageBox, QProgressBar, QScrollArea,
    QFrame, QSplitter, QScrollBar, QDialog, QDialogButtonBox,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QSizePolicy, QTextEdit,
    QGroupBox, QSpacerItem,
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QTimer, QRect, QPoint,
)
from PyQt5.QtGui import (
    QPixmap, QImage, QFont, QColor, QPainter, QPen, QBrush,
    QFontMetrics, QCursor, QPalette,
)

# ── Deferred heavy imports (loaded after splash) ────────────────────
pd = None
np = None
cv2 = None

# ── Colour palette (matches original pastel green/yellow theme) ─────
BG_MAIN     = "#f5f7f0"
BG_BLOCK    = "#eafaf1"
BORDER      = "#bbded6"
GREEN_DARK  = "#388659"
GREEN_MED   = "#21a179"
GREEN_LIGHT = "#d2f8d2"
BLUE_DARK   = "#406882"
BLUE_LIGHT  = "#1eb2e6"
YELLOW      = "#fae588"
TEXT_GREEN  = "#457267"
TEXT_DARK   = "#222222"
WHITE       = "#ffffff"
PATH_BG     = "#f7fff7"
PROGRESS_BLUE = "#00589C"


# ── Stylesheet ──────────────────────────────────────────────────────
GLOBAL_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: transparent;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 10pt;
    color: {TEXT_DARK};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: rgba(245, 247, 240, 210);
}}
QTabBar::tab {{
    background: #daf0e8;
    color: {GREEN_DARK};
    padding: 0.5em 1.4em;
    border: 1px solid {BORDER};
    border-bottom: none;
    font-weight: bold;
}}
QTabBar::tab:selected {{
    background: {BG_BLOCK};
    color: {BLUE_DARK};
}}
QPushButton {{
    border-radius: 6px;
    padding: 0.4em 0.9em;
    font-weight: bold;
    color: white;
    background-color: {GREEN_MED};
    border: none;
}}
QPushButton:hover {{
    opacity: 0.85;
}}
QPushButton:pressed {{
    background-color: {GREEN_DARK};
}}
QLineEdit {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 0.2em 0.4em;
}}
QCheckBox {{
    spacing: 0.4em;
    font-size: 10pt;
}}
QCheckBox::indicator {{
    width: 1em;
    height: 1em;
}}
QCheckBox::indicator:unchecked {{
    border: 1px solid #aaa;
    border-radius: 3px;
    background: white;
}}
QCheckBox::indicator:checked {{
    border: 1px solid {GREEN_DARK};
    border-radius: 3px;
    background: {GREEN_MED};
}}
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 5px;
    background: #dcdcdc;
    min-height: 12px;
    max-height: 18px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {PROGRESS_BLUE};
    border-radius: 4px;
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: #dcdcdc;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {GREEN_MED};
    border: 1px solid {GREEN_DARK};
    width: 14px;
    height: 14px;
    border-radius: 7px;
    margin: -4px 0;
}}
QSlider::sub-page:horizontal {{
    background: {GREEN_MED};
    border-radius: 3px;
}}
QScrollBar:horizontal {{
    height: 10px;
    background: #f0f0f0;
}}
QScrollBar::handle:horizontal {{
    background: #aaa;
    border-radius: 5px;
    min-width: 20px;
}}
QLabel#path_label {{
    background: {PATH_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT_GREEN};
    padding: 0.2em 0.4em;
    font-size: 9pt;
}}
QGroupBox {{
    background: {BG_BLOCK};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 0.6em;
    padding: 0.5em;
    font-weight: bold;
    font-size: 10pt;
    color: {GREEN_DARK};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QTreeWidget {{
    border: 1px solid {BORDER};
    background: white;
    alternate-background-color: #f0fdf4;
    font-size: 9pt;
}}
QHeaderView::section {{
    background: {BG_BLOCK};
    padding: 4px;
    border: 1px solid {BORDER};
    font-weight: bold;
    font-size: 9pt;
}}
"""


def resource_path(relative: str) -> Path:
    """Resolve resource paths for both development and PyInstaller builds."""
    try:
        base = Path(sys._MEIPASS)
    except AttributeError:
        base = Path(os.path.abspath("."))
    return base / relative


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════
# Worker thread base
# ═══════════════════════════════════════════════════════════════════
class WorkerThread(QThread):
    progress = pyqtSignal(float)   # 0.0 – 1.0
    finished = pyqtSignal(str)     # success message
    error    = pyqtSignal(str)     # error message


# ═══════════════════════════════════════════════════════════════════
# Custom Class Names Popup Dialog
# ═══════════════════════════════════════════════════════════════════
class CustomClassNamesDialog(QDialog):
    """Popup for setting custom class name strings for each of the 13 classes.

    Names are ordered by YOLO class ID (0–12):
      ID 0→C2, 1→C5, 2→C1, 3→C3-B, 4→C4, 5→C11, 6→C9,
         7→C2-B, 8→C8, 9→C6, 10→C7-E, 11→C7-F, 12→C7-G
    This matches the order used by draw_yolo_overlays and the label files.
    """

    # Ordered by YOLO class ID 0–12
    DEFAULT_NAMES = ["C2", "C5", "C1", "C3-B", "C4", "C11", "C9",
                     "C2-B", "C8", "C6", "C7-E", "C7-F", "C7-G"]

    def __init__(self, parent=None, current_names=None, enabled=False):
        super().__init__(parent)
        self.setWindowTitle("Custom Class Names")
        self.setMinimumWidth(420)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        layout = QVBoxLayout(self)

        # Enable toggle
        self.enable_cb = QCheckBox("Enable custom class names")
        self.enable_cb.setChecked(enabled)
        self.enable_cb.setStyleSheet(f"font-weight: bold; color: {GREEN_DARK};")
        self.enable_cb.toggled.connect(self._on_toggle)
        layout.addWidget(self.enable_cb)

        info = QLabel(
            "If enabled: 'Save images' is forced OFF for the prediction run.\n"
            "Custom names are used in all outputs: CSV columns, chart labels,\n"
            "and bounding-box overlays. Names below are listed by YOLO class ID\n"
            "(the order the model uses internally in its label files)."
        )
        info.setStyleSheet(f"color: {BLUE_DARK}; font-size: 11px; padding: 4px 0;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Grid of name entries
        grid = QGridLayout()
        grid.setSpacing(6)
        self.entries = []
        init_names = current_names if current_names else self.DEFAULT_NAMES[:]
        for i, (dname, cname) in enumerate(zip(self.DEFAULT_NAMES, init_names)):
            row = i // 2
            col_base = (i % 2) * 3
            grid.addWidget(QLabel(f"{dname}:"), row, col_base)
            entry = QLineEdit(cname)
            entry.setMinimumWidth(90)
            entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            entry.setToolTip(f"Custom label for class {dname}")
            grid.addWidget(entry, row, col_base + 1)
            self.entries.append(entry)
        layout.addLayout(grid)

        # Buttons
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

        self._on_toggle(enabled)

    def _on_toggle(self, state):
        for e in self.entries:
            e.setEnabled(state)

    def get_values(self):
        """Return (enabled: bool, names: list[str])"""
        names = [e.text().strip() or self.DEFAULT_NAMES[i]
                 for i, e in enumerate(self.entries)]
        return self.enable_cb.isChecked(), names


# ═══════════════════════════════════════════════════════════════════
# Class Reference Popup
# ═══════════════════════════════════════════════════════════════════
class ClassReferenceDialog(QDialog):
    def __init__(self, parent=None, img_path=None):
        super().__init__(parent)
        self.setWindowTitle("Morphotype Class Reference")
        self.resize(800, 600)
        self.setStyleSheet(GLOBAL_STYLESHEET)
        layout = QVBoxLayout(self)
        self._lbl = QLabel(alignment=Qt.AlignCenter)
        self._lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._lbl)
        self._orig_pixmap = None
        if img_path and Path(img_path).exists():
            self._orig_pixmap = QPixmap(str(img_path))
            self._update_image()
        else:
            self._lbl.setText("Reference image not found.")
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        btn.setStyleSheet(f"background:{GREEN_MED}; color:white; padding:6px 20px;")
        layout.addWidget(btn, alignment=Qt.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_image()

    def _update_image(self):
        if self._orig_pixmap:
            scaled = self._orig_pixmap.scaled(
                self._lbl.width() - 20, self._lbl.height() - 20,
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._lbl.setPixmap(scaled)


# ═══════════════════════════════════════════════════════════════════
# CSV Table Dialog
# ═══════════════════════════════════════════════════════════════════
class CSVTableDialog(QDialog):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.setWindowTitle("CSV Data Table")
        self.resize(1000, 600)
        self.setStyleSheet(GLOBAL_STYLESHEET)
        self._df = df
        layout = QVBoxLayout(self)
        tree = QTreeWidget()
        tree.setHeaderLabels(list(df.columns))
        tree.header().setSectionResizeMode(QHeaderView.Interactive)
        tree.setAlternatingRowColors(True)
        for _, row in df.iterrows():
            item = QTreeWidgetItem([str(v) for v in row])
            tree.addTopLevelItem(item)
        layout.addWidget(tree)
        btn = QPushButton("Export CSV")
        btn.setStyleSheet(f"background:{BLUE_DARK}; color:white;")
        btn.clicked.connect(self._export)
        layout.addWidget(btn, alignment=Qt.AlignRight)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv)")
        if path:
            self._df.to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"CSV exported to:\n{path}")


# ═══════════════════════════════════════════════════════════════════
# Bounding Box Overlay (pure cv2, uses YOLO .txt labels)
# ═══════════════════════════════════════════════════════════════════
def draw_yolo_overlays(image_dir: Path, label_dir: Path, output_dir: Path,
                       custom_names: list, class_color_map: dict = None):
    """
    Read YOLO label .txt files from label_dir, draw bounding boxes with
    custom_names on original images from image_dir, write annotated copies
    to output_dir. Does NOT rely on ultralytics rendering.

    Args:
        image_dir:       folder with original images
        label_dir:       folder with YOLO .txt label files
        output_dir:      destination folder for annotated images
        custom_names:    list of 13 strings, indexed by YOLO class id
        class_color_map: optional dict class_id->BGR tuple; auto-generated if None
    """
    import cv2 as _cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a colour palette (one per class)
    NUM_CLASSES = len(custom_names)
    if class_color_map is None:
        import colorsys
        class_color_map = {}
        for cid in range(NUM_CLASSES):
            h = cid / max(NUM_CLASSES, 1)
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
            class_color_map[cid] = (int(b * 255), int(g * 255), int(r * 255))

    label_files = list(label_dir.glob("*.txt"))
    if not label_files:
        return 0

    written = 0
    for lf in label_files:
        # Find matching image (any supported extension)
        img_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = image_dir / (lf.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        img = _cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        with open(lf, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                if cid < 0 or cid >= NUM_CLASSES:
                    continue

                x1 = int((cx - bw / 2) * iw)
                y1 = int((cy - bh / 2) * ih)
                x2 = int((cx + bw / 2) * iw)
                y2 = int((cy + bh / 2) * ih)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw - 1, x2), min(ih - 1, y2)

                color = class_color_map.get(cid, (0, 255, 0))
                _cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label_text = custom_names[cid] if cid < len(custom_names) else str(cid)
                font_scale = max(0.4, min(iw, ih) / 1000)
                thickness = max(1, int(font_scale * 2))
                (tw, th), baseline = _cv2.getTextSize(
                    label_text, _cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                ty = max(y1 - 4, th + 4)
                _cv2.rectangle(img,
                               (x1, ty - th - baseline - 2),
                               (x1 + tw + 4, ty + baseline - 2),
                               color, -1)
                _cv2.putText(img, label_text, (x1 + 2, ty - 2),
                             _cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                             (255, 255, 255), thickness, _cv2.LINE_AA)

        out_path = output_dir / img_path.name
        _cv2.imwrite(str(out_path), img)
        written += 1

    return written


# ═══════════════════════════════════════════════════════════════════
# Image Viewer Widget (replaces tkinter canvas-based viewer)
# ═══════════════════════════════════════════════════════════════════
class ZoomPanLabel(QLabel):
    """Image label with mouse-wheel zoom and click-drag pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background: white;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pil_img  = None
        self._zoom     = 1.0
        self._offset   = QPoint(0, 0)
        self._drag_pos = None
        self.setMouseTracking(True)

    def set_pil_image(self, pil_img):
        self._pil_img = pil_img
        self._fit_zoom()
        self._offset = QPoint(0, 0)
        self._render()

    def _fit_zoom(self):
        if self._pil_img is None:
            return
        iw, ih = self._pil_img.size
        sw, sh = self.width(), self.height()
        if iw == 0 or ih == 0:
            return
        self._zoom = min(sw / iw, sh / ih)

    def _render(self):
        if self._pil_img is None:
            self.clear()
            return
        iw, ih = self._pil_img.size
        sw, sh = max(self.width(), 1), max(self.height(), 1)
        nw = max(1, int(iw * self._zoom))
        nh = max(1, int(ih * self._zoom))
        resized = self._pil_img.resize((nw, nh), 1)  # LANCZOS=1 in Pillow>=10

        data = resized.convert("RGBA").tobytes("raw", "RGBA")
        qimg = QImage(data, nw, nh, QImage.Format_RGBA8888)
        pm = QPixmap.fromImage(qimg)

        canvas = QPixmap(sw, sh)
        canvas.fill(QColor("white"))
        painter = QPainter(canvas)

        # Clamped offset
        ox = self._offset.x()
        oy = self._offset.y()
        if nw <= sw:
            ox = (sw - nw) // 2
            self._offset.setX(0)
        else:
            ox = max(0, min(ox, nw - sw))
            self._offset.setX(ox)
            ox = -ox
        if nh <= sh:
            oy = (sh - nh) // 2
            self._offset.setY(0)
        else:
            oy = max(0, min(oy, nh - sh))
            self._offset.setY(oy)
            oy = -oy

        painter.drawPixmap(ox, oy, pm)

        # Zoom label
        fit = min(sw / iw, sh / ih) if iw and ih else 1.0
        painter.setPen(QColor(GREEN_DARK))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(4, sh - 4, f"Zoom: {self._zoom / fit:.2f}x")
        painter.end()
        self.setPixmap(canvas)

    def wheelEvent(self, event):
        if self._pil_img is None:
            return
        iw, ih = self._pil_img.size
        sw, sh = max(self.width(), 1), max(self.height(), 1)
        fit = min(sw / iw, sh / ih) if iw and ih else 1.0
        delta = event.angleDelta().y()
        factor = 1.13 if delta > 0 else 0.88
        new_zoom = max(fit, min(self._zoom * factor, fit * 10))
        # zoom towards mouse position
        mx = event.pos().x()
        my = event.pos().y()
        ox = self._offset.x()
        oy = self._offset.y()
        rel_x = (ox + mx) / self._zoom
        rel_y = (oy + my) / self._zoom
        self._offset.setX(int(rel_x * new_zoom - mx))
        self._offset.setY(int(rel_y * new_zoom - my))
        self._zoom = new_zoom
        self._render()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            delta = event.pos() - self._drag_pos
            self._offset -= delta
            self._drag_pos = event.pos()
            self._render()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()


# ═══════════════════════════════════════════════════════════════════
# Background-image central widget
# ═══════════════════════════════════════════════════════════════════
class BackgroundWidget(QWidget):
    """A QWidget that tiles/stretches a PNG image as its background.

    Drop this in as the central widget and all child widgets sit on
    top of the image automatically.

    Usage
    -----
    Set the path to your PNG via ``set_background(path)``.  The image
    is scaled to fill the widget while keeping its aspect ratio
    (Qt.KeepAspectRatioByExpanding), so it always covers the full
    window without distortion.  If you prefer a plain stretch, swap
    the AspectRatioMode for Qt.IgnoreAspectRatio.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg_pixmap: QPixmap | None = None

    def set_background(self, path: str) -> bool:
        """Load a PNG (or any Qt-supported image) as the background.

        Returns True on success, False if the file could not be loaded.
        """
        px = QPixmap(path)
        if px.isNull():
            return False
        self._bg_pixmap = px
        self.update()          # trigger a repaint
        return True

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._bg_pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        scaled = self._bg_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        # centre the (potentially oversized) scaled image
        x = (self.width()  - scaled.width())  // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)


# ═══════════════════════════════════════════════════════════════════
# Main Application Window
# ═══════════════════════════════════════════════════════════════════
class MycoVisionApp(QMainWindow):

    # Canonical display order of 13 classes — used for CSV columns, plots, checkboxes
    CLASS_NAMES_ORDERED = ["C1", "C2", "C2-B", "C3-B", "C5", "C8", "C9",
                           "C4", "C6", "C11", "C7-E", "C7-F", "C7-G"]
    # YOLO model class-ID order (id 0–12) — used for label file parsing and overlays
    YOLO_CLASS_NAMES    = ["C2", "C5", "C1", "C3-B", "C4", "C11", "C9",
                           "C2-B", "C8", "C6", "C7-E", "C7-F", "C7-G"]
    # YOLO class id → index in CLASS_NAMES_ORDERED
    YOLO_TO_CLASS_INDEX = [2, 0, 7, 8, 1, 9, 6, 3, 4, 5, 10, 11, 12]
    # Display order for checkboxes (matches original UI)
    CLASS_DISPLAY_ORDER = ["C2", "C5", "C1", "C4", "C6", "C11",
                           "C9", "C2-B", "C3-B", "C8", "C7-E", "C7-F", "C7-G"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TU_MyCo-Vision: Fungal Morphology Detector")
        self.resize(1300, 900)
        self.setMinimumSize(900, 650)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # ── state ──────────────────────────────────────────────────
        self.input_folder   = ""
        self.output_folder  = ""
        self.confidence     = 0.20
        self.line_width     = 2
        self.agnostic_nms   = False
        self.show_output    = False
        self.analysis_name  = "Myco_Analysis"

        # 13 checkboxes (selected classes for detection)
        self.selected_classes = [False] * 13
        # 13 checkboxes (excluded classes for analysis)
        self.excluded_classes = [False] * 13

        # Custom class names feature
        self.custom_names_enabled = False
        self.custom_class_names   = self.CLASS_NAMES_ORDERED[:]  # 13 entries

        # Analysis
        self.single_group_folder           = ""
        self.single_analysis_name          = "SingleAnalysis"
        self.single_analysis_output_folder = ""
        self.multi_analysis_name           = "MultiAnalysis"
        self.multi_analysis_output_folder  = ""
        self.group_folders                 = []
        self.last_analysis_csv             = None
        self.latest_prediction_folder      = None

        # Dashboard
        self.orig_img_folder = ""
        self.pred_img_folder = ""
        self._dashboard_matches   = []
        self._dashboard_orig_map  = {}
        self._dashboard_pred_map  = {}
        self._dashboard_idx       = 0

        # ── Splash ─────────────────────────────────────────────────
        self._show_splash()

    # ──────────────────────────────────────────────────────────────
    # Splash Screen
    # ──────────────────────────────────────────────────────────────
    def _show_splash(self):
        central = BackgroundWidget()
        # ── background image ──────────────────────────────────────────
        # Place your PNG next to the script (or update the path below).
        # Safe to leave in even when the file does not exist yet.
        central.set_background(str(resource_path("aureo1.png")))
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setAlignment(Qt.AlignCenter)

        splash = QFrame()
        splash.setFixedSize(380, 160)
        splash.setStyleSheet(
            f"background:{WHITE}; border-radius:18px; "
            f"border:1px solid {BORDER};"
        )
        s_lay = QVBoxLayout(splash)
        lbl = QLabel("✨ Loading TU_MyCo-Vision...\nPlease wait...")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            f"color:{GREEN_MED}; font-size:11pt; font-weight:bold; border:none;"
        )
        s_lay.addWidget(lbl)
        self._splash_progress = QProgressBar()
        self._splash_progress.setRange(0, 100)
        self._splash_progress.setValue(10)
        self._splash_progress.setStyleSheet(
            f"QProgressBar::chunk {{ background:{GREEN_MED}; }}"
        )
        s_lay.addWidget(self._splash_progress)
        lay.addWidget(splash)

        # start heavy imports in a thread
        self._import_thread = _ImportThread()
        self._import_thread.done.connect(self._on_imports_done)
        self._import_thread.start()

        self._splash_timer = QTimer(self)
        self._splash_timer.timeout.connect(self._bump_splash)
        self._splash_timer.start(200)

    def _bump_splash(self):
        v = self._splash_progress.value()
        if v < 90:
            self._splash_progress.setValue(v + 8)

    def _on_imports_done(self):
        self._splash_timer.stop()
        self._splash_progress.setValue(100)
        QTimer.singleShot(400, self._build_main_ui)

    # ──────────────────────────────────────────────────────────────
    # Main UI
    # ──────────────────────────────────────────────────────────────
    def _build_main_ui(self):
        central = BackgroundWidget()
        central.set_background(str(resource_path("background.png")))
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(7, 6, 7, 6)

        # shared task progress bar (top of window)
        self.task_progress = QProgressBar()
        self.task_progress.setRange(0, 100)
        self.task_progress.setValue(0)
        self.task_progress.setFixedHeight(15)
        layout.addWidget(self.task_progress)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        detection_page = QWidget()
        analysis_page  = QWidget()
        results_page   = QWidget()
        self.tabs.addTab(detection_page, "Detection")
        self.tabs.addTab(analysis_page,  "Analysis")
        self.tabs.addTab(results_page,   "Results Dashboard")

        self._build_detection_tab(detection_page)
        self._build_analysis_tab(analysis_page)
        self._build_results_tab(results_page)

    def _set_progress(self, fraction: float):
        """Thread-safe progress update."""
        QTimer.singleShot(0, lambda: self.task_progress.setValue(int(fraction * 100)))

    # ══════════════════════════════════════════════════════════════
    # Detection Tab
    # ══════════════════════════════════════════════════════════════
    def _build_detection_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # ── 1. File I/O Block ──────────────────────────────────────
        io_group = QGroupBox("Select Image & Output Folder")
        io_group.setStyleSheet(
            f"QGroupBox {{ color:{GREEN_DARK}; font-size:12pt; }}"
        )
        io_lay = QGridLayout(io_group)
        io_lay.setSpacing(8)

        btn_img = self._btn("Image Folder", GREEN_MED)
        btn_img.clicked.connect(self.browse_input)
        btn_out = self._btn("Output Folder", BLUE_DARK)
        btn_out.clicked.connect(self.browse_output)
        self._input_path_lbl  = self._path_label("")
        self._output_path_lbl = self._path_label("")
        io_lay.addWidget(btn_img,                0, 0)
        io_lay.addWidget(btn_out,                0, 1)
        io_lay.addWidget(self._input_path_lbl,   1, 0)
        io_lay.addWidget(self._output_path_lbl,  1, 1)
        io_lay.setColumnStretch(0, 1)
        io_lay.setColumnStretch(1, 1)
        layout.addWidget(io_group)

        # ── 2. Middle row: Classes + Model Settings ────────────────
        mid_row = QHBoxLayout()
        mid_row.setSpacing(10)

        # ── 2a. Select Classes block ───────────────────────────────
        class_group = QGroupBox("Select Classes")
        class_group.setStyleSheet(
            f"QGroupBox {{ color:{GREEN_DARK}; font-size:12pt; }}"
        )
        class_lay = QGridLayout(class_group)
        class_lay.setSpacing(6)

        btn_guide = self._btn("ℹ️ Open Class Guide", YELLOW,
                              text_color=GREEN_DARK, font_size=10)
        btn_guide.clicked.connect(self.show_class_reference_popup)
        class_lay.addWidget(btn_guide, 0, 0, alignment=Qt.AlignLeft)

        btn_all   = self._btn("Select All",      GREEN_LIGHT, text_color=GREEN_DARK)
        btn_clear = self._btn("Clear Selection", YELLOW,      text_color=GREEN_DARK)
        btn_custom = self._btn("Custom class names…", BLUE_DARK)
        btn_all.clicked.connect(self.select_all_classes)
        btn_clear.clicked.connect(self.clear_classes)
        btn_custom.clicked.connect(self.show_custom_class_names_popup)
        class_lay.addWidget(btn_all,    1, 0)
        class_lay.addWidget(btn_clear,  1, 1)
        class_lay.addWidget(btn_custom, 1, 2, 1, 2)

        self._class_checkboxes = []
        for i, cname in enumerate(self.CLASS_DISPLAY_ORDER):
            cb = QCheckBox(cname)
            cb.setChecked(self.selected_classes[i])
            cb.stateChanged.connect(
                lambda state, idx=i: self._on_class_checked(idx, state))
            row_i = 2 + i // 4
            col_i = i % 4
            class_lay.addWidget(cb, row_i, col_i)
            self._class_checkboxes.append(cb)

        # Custom names indicator label
        self._custom_names_indicator = QLabel("")
        self._custom_names_indicator.setStyleSheet(
            f"color:{BLUE_DARK}; font-size:8pt; font-style:italic;")
        class_lay.addWidget(self._custom_names_indicator, 5, 0, 1, 4)

        mid_row.addWidget(class_group, stretch=1)

        # ── 2b. Model Settings ─────────────────────────────────────
        model_group = QGroupBox("Model Settings")
        model_group.setStyleSheet(
            f"QGroupBox {{ color:{BLUE_DARK}; font-size:12pt; }}"
        )
        model_lay = QGridLayout(model_group)
        model_lay.setSpacing(6)

        model_lay.addWidget(QLabel("Confidence:"), 0, 0, Qt.AlignRight)
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(10, 100)
        self._conf_slider.setValue(int(self.confidence * 100))
        self._conf_slider.setTickInterval(5)
        self._conf_slider.valueChanged.connect(self._on_conf_changed)
        model_lay.addWidget(self._conf_slider, 0, 1)
        self._conf_lbl = QLabel(f"{self.confidence:.2f}")
        self._conf_lbl.setStyleSheet("font-weight:bold;")
        model_lay.addWidget(self._conf_lbl, 1, 1, Qt.AlignLeft)

        model_lay.addWidget(QLabel("Line Width:"), 2, 0, Qt.AlignRight)
        self._line_width_entry = QLineEdit(str(self.line_width))
        self._line_width_entry.setMinimumWidth(40)
        self._line_width_entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        model_lay.addWidget(self._line_width_entry, 2, 1, Qt.AlignLeft)

        model_lay.addWidget(QLabel("Analysis Name:"), 3, 0, Qt.AlignRight)
        self._analysis_name_entry = QLineEdit(self.analysis_name)
        self._analysis_name_entry.setMinimumWidth(110)
        self._analysis_name_entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        model_lay.addWidget(self._analysis_name_entry, 3, 1, Qt.AlignLeft)

        self._agnostic_cb = QCheckBox("Agnostic NMS")
        self._agnostic_cb.setChecked(self.agnostic_nms)
        model_lay.addWidget(self._agnostic_cb, 4, 0, Qt.AlignLeft)

        self._show_output_cb = QCheckBox("Show Output")
        self._show_output_cb.setChecked(self.show_output)
        model_lay.addWidget(self._show_output_cb, 4, 1, Qt.AlignLeft)

        mid_row.addWidget(model_group, stretch=1)
        layout.addLayout(mid_row)

        # ── 3. Predict Button ──────────────────────────────────────
        btn_pred = self._btn(
            "▶  Run MyCo-Vision Prediction",
            GREEN_MED, font_size=18, height=60
        )
        btn_pred.clicked.connect(self.run_prediction_threaded)
        layout.addWidget(btn_pred)

    # ── Detection helpers ──────────────────────────────────────────
    def _on_class_checked(self, idx, state):
        self.selected_classes[idx] = (state == Qt.Checked)

    def _on_conf_changed(self, value):
        self.confidence = value / 100.0
        self._conf_lbl.setText(f"{self.confidence:.2f}")

    def browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if path:
            self.input_folder = path
            self._input_path_lbl.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_folder = path
            self._output_path_lbl.setText(path)

    def select_all_classes(self):
        for cb in self._class_checkboxes:
            cb.setChecked(True)

    def clear_classes(self):
        for cb in self._class_checkboxes:
            cb.setChecked(False)

    def show_class_reference_popup(self):
        img_path = resource_path("reference.png")
        dlg = ClassReferenceDialog(self, str(img_path))
        dlg.exec_()

    def show_custom_class_names_popup(self):
        # Pass current custom names to dialog converted to YOLO order for display
        yolo_current = _canonical_to_yolo_names(self.custom_class_names)
        dlg = CustomClassNamesDialog(
            self,
            current_names=yolo_current,
            enabled=self.custom_names_enabled
        )
        if dlg.exec_() == QDialog.Accepted:
            enabled, yolo_names = dlg.get_values()
            self.custom_names_enabled = enabled
            # Store internally as canonical order (for CSV/plot pipelines)
            self.custom_class_names = _yolo_to_canonical_names(yolo_names)
            if self.custom_names_enabled:
                self._custom_names_indicator.setText(
                    "✔ Custom class names active — Save Images forced OFF"
                )
            else:
                self._custom_names_indicator.setText("")

    def run_prediction_threaded(self):
        # Collect current UI values
        self.input_folder   = self.input_folder
        self.output_folder  = self.output_folder
        self.analysis_name  = self._analysis_name_entry.text().strip() or "Myco_Analysis"
        self.line_width     = int(self._line_width_entry.text() or "2")
        self.agnostic_nms   = self._agnostic_cb.isChecked()
        self.show_output    = self._show_output_cb.isChecked()

        selected_cls = [i for i, v in enumerate(self.selected_classes) if v]
        if not self.input_folder:
            QMessageBox.critical(self, "Input Error", "Please select an input folder.")
            return
        if not self.output_folder:
            QMessageBox.critical(self, "Input Error", "Please select an output folder.")
            return
        if not selected_cls:
            QMessageBox.critical(self, "Input Error", "Please select at least one class.")
            return

        # If custom names enabled → force save=False (we'll overlay ourselves)
        save_images = not self.custom_names_enabled

        self._pred_thread = PredictionThread(
            input_folder    = self.input_folder,
            output_folder   = self.output_folder,
            confidence      = self.confidence,
            line_width      = self.line_width,
            selected_cls    = selected_cls,
            agnostic_nms    = self.agnostic_nms,
            show_output     = self.show_output,
            analysis_name   = self.analysis_name,
            save_images     = save_images,
            custom_names_enabled = self.custom_names_enabled,
            custom_class_names   = self.custom_class_names,
        )
        self._pred_thread.progress.connect(self._set_progress)
        self._pred_thread.finished.connect(self._on_pred_done)
        self._pred_thread.error.connect(self._on_pred_error)
        self._pred_thread.start()

    def _on_pred_done(self, msg):
        self._set_progress(0.0)
        QMessageBox.information(self, "Done", msg)

    def _on_pred_error(self, msg):
        self._set_progress(0.0)
        QMessageBox.critical(self, "Prediction Error", msg)

    # ══════════════════════════════════════════════════════════════
    # Analysis Tab
    # ══════════════════════════════════════════════════════════════
    def _build_analysis_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # ── Class Exclusion block ──────────────────────────────────
        excl_group = QGroupBox(
            "Class Exclusion (Exclude & Merge Selected Classes as 'Others')"
        )
        excl_group.setStyleSheet(
            f"QGroupBox {{ color:{GREEN_DARK}; font-size:11pt; }}"
        )
        excl_lay = QGridLayout(excl_group)
        excl_lay.setSpacing(6)

        btn_clear_excl = self._btn("Clear Selection", YELLOW, text_color=GREEN_DARK)
        btn_clear_excl.clicked.connect(self.clear_excluded_classes)
        excl_lay.addWidget(btn_clear_excl, 0, 3, Qt.AlignRight)

        self._excl_checkboxes = []
        for i, cname in enumerate(self.CLASS_NAMES_ORDERED):
            cb = QCheckBox(cname)
            cb.setChecked(self.excluded_classes[i])
            cb.stateChanged.connect(
                lambda state, idx=i: self._on_excl_checked(idx, state))
            row_i = 1 + i // 4
            col_i = i % 4
            excl_lay.addWidget(cb, row_i, col_i)
            self._excl_checkboxes.append(cb)
        layout.addWidget(excl_group)

        # ── Single + Multi Group side by side ──────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        # ── Single Group Analysis ──────────────────────────────────
        sg_group = QGroupBox("Single Group Analysis")
        sg_group.setStyleSheet(
            f"QGroupBox {{ color:{GREEN_DARK}; font-size:10pt; }}"
        )
        sg_lay = QGridLayout(sg_group)
        sg_lay.setSpacing(6)

        btn_sg_folder = self._btn("Select Folder", GREEN_MED)
        btn_sg_folder.clicked.connect(self.browse_single_group_folder)
        self._sg_folder_lbl = self._path_label("No folder selected")
        sg_lay.addWidget(btn_sg_folder,        0, 0)
        sg_lay.addWidget(self._sg_folder_lbl,  0, 1)

        btn_sg_out = self._btn("Select Output Folder", BLUE_DARK)
        btn_sg_out.clicked.connect(self.select_single_analysis_output_folder)
        self._sg_out_lbl = self._path_label("No folder selected")
        sg_lay.addWidget(btn_sg_out,       1, 0)
        sg_lay.addWidget(self._sg_out_lbl, 1, 1)

        sg_lay.addWidget(QLabel("Analysis Name:"), 2, 0, Qt.AlignRight)
        self._sg_name_entry = QLineEdit(self.single_analysis_name)
        sg_lay.addWidget(self._sg_name_entry, 2, 1, Qt.AlignLeft)

        btn_run_sg = self._btn("Run Single Group Analysis", GREEN_DARK,
                               font_size=13, height=40)
        btn_run_sg.clicked.connect(self.run_single_group_analysis_threaded)
        sg_lay.addWidget(btn_run_sg, 3, 0, 1, 2)
        sg_lay.setColumnStretch(1, 1)
        bottom_row.addWidget(sg_group)

        # ── Multi Group Analysis ───────────────────────────────────
        mg_group = QGroupBox("Multi Group Analysis")
        mg_group.setStyleSheet(
            f"QGroupBox {{ color:{BLUE_DARK}; font-size:10pt; }}"
        )
        mg_lay = QGridLayout(mg_group)
        mg_lay.setSpacing(6)

        btn_add_group   = self._btn("Add Group Folders", GREEN_MED)
        btn_clear_group = self._btn("Clear Groups", GREEN_LIGHT, text_color=GREEN_DARK)
        btn_add_group.clicked.connect(self.add_group_folder)
        btn_clear_group.clicked.connect(self.clear_group_folders)
        mg_lay.addWidget(btn_add_group,   0, 0)
        mg_lay.addWidget(btn_clear_group, 0, 1)

        mg_lay.addWidget(QLabel("Selected Groups:"), 1, 0, Qt.AlignTop | Qt.AlignRight)
        self._mg_groups_lbl = self._path_label("No groups selected")
        self._mg_groups_lbl.setWordWrap(True)
        mg_lay.addWidget(self._mg_groups_lbl, 1, 1)

        btn_mg_out = self._btn("Select Output Folder", BLUE_DARK)
        btn_mg_out.clicked.connect(self.select_multi_analysis_output_folder)
        self._mg_out_lbl = self._path_label("No folder selected")
        mg_lay.addWidget(btn_mg_out,       2, 0)
        mg_lay.addWidget(self._mg_out_lbl, 2, 1)

        mg_lay.addWidget(QLabel("Analysis Name:"), 3, 0, Qt.AlignRight)
        self._mg_name_entry = QLineEdit(self.multi_analysis_name)
        mg_lay.addWidget(self._mg_name_entry, 3, 1, Qt.AlignLeft)

        btn_run_mg = self._btn("Run Multi Group Analysis", BLUE_DARK,
                               font_size=13, height=40)
        btn_run_mg.clicked.connect(self.run_multi_group_analysis_threaded)
        mg_lay.addWidget(btn_run_mg, 4, 0, 1, 2)
        mg_lay.setColumnStretch(1, 1)
        bottom_row.addWidget(mg_group)

        layout.addLayout(bottom_row)
        layout.addStretch()

    # ── Analysis helpers ───────────────────────────────────────────
    def _on_excl_checked(self, idx, state):
        self.excluded_classes[idx] = (state == Qt.Checked)

    def clear_excluded_classes(self):
        for cb in self._excl_checkboxes:
            cb.setChecked(False)

    def browse_single_group_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Folder for Single Group Analysis")
        if path:
            self.single_group_folder = path
            self._sg_folder_lbl.setText(path)

    def select_single_analysis_output_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Single Group Analysis")
        if path:
            self.single_analysis_output_folder = path
            self._sg_out_lbl.setText(path)

    def add_group_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Add Group Folder for Multi-Group Analysis")
        if path and path not in self.group_folders:
            self.group_folders.append(path)
            self._mg_groups_lbl.setText(
                "\n".join(Path(p).name for p in self.group_folders))

    def clear_group_folders(self):
        self.group_folders = []
        self._mg_groups_lbl.setText("No groups selected")

    def select_multi_analysis_output_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Multi Group Analysis")
        if path:
            self.multi_analysis_output_folder = path
            self._mg_out_lbl.setText(path)

    def run_single_group_analysis_threaded(self):
        self.single_analysis_name = self._sg_name_entry.text().strip() or "SingleAnalysis"
        folder = self.single_group_folder
        if not folder or not Path(folder).exists():
            QMessageBox.critical(self, "Error", "Please select a valid folder.")
            return
        save_dir = self.single_analysis_output_folder or self.output_folder or folder
        excluded  = [i for i, v in enumerate(self.excluded_classes) if v]
        custom_names = self.custom_class_names if self.custom_names_enabled else None

        self._sg_thread = SingleGroupAnalysisThread(
            folder=folder,
            analysis_name=self.single_analysis_name,
            save_dir=save_dir,
            excluded_indices=excluded,
            custom_names=custom_names,
        )
        self._sg_thread.progress.connect(self._set_progress)
        self._sg_thread.finished.connect(self._on_analysis_done)
        self._sg_thread.error.connect(self._on_analysis_error)
        self._sg_thread.csv_ready.connect(self._on_csv_ready)
        self._sg_thread.start()

    def run_multi_group_analysis_threaded(self):
        self.multi_analysis_name = self._mg_name_entry.text().strip() or "MultiAnalysis"
        if not self.group_folders:
            QMessageBox.critical(self, "Error", "Please add at least one group folder.")
            return
        save_dir = self.multi_analysis_output_folder or self.output_folder or self.group_folders[0]
        excluded  = [i for i, v in enumerate(self.excluded_classes) if v]
        custom_names = self.custom_class_names if self.custom_names_enabled else None

        self._mg_thread = MultiGroupAnalysisThread(
            folders=self.group_folders,
            analysis_name=self.multi_analysis_name,
            save_dir=save_dir,
            excluded_indices=excluded,
            custom_names=custom_names,
        )
        self._mg_thread.progress.connect(self._set_progress)
        self._mg_thread.finished.connect(self._on_analysis_done)
        self._mg_thread.error.connect(self._on_analysis_error)
        self._mg_thread.csv_ready.connect(self._on_csv_ready)
        self._mg_thread.start()

    def _on_analysis_done(self, msg):
        self._set_progress(0.0)
        QMessageBox.information(self, "Done", msg)

    def _on_analysis_error(self, msg):
        self._set_progress(0.0)
        QMessageBox.critical(self, "Analysis Error", msg)

    def _on_csv_ready(self, csv_path):
        self.last_analysis_csv = csv_path

    # ══════════════════════════════════════════════════════════════
    # Results Dashboard Tab
    # ══════════════════════════════════════════════════════════════
    def _build_results_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Button row ─────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_load  = self._btn("Load Analysis CSV", GREEN_DARK)
        btn_table = self._btn("View CSV Table",    BLUE_DARK)
        btn_plots = self._btn("Click to View Plots", "#B8860B")
        btn_load.clicked.connect(self.dashboard_load_csv)
        btn_table.clicked.connect(self.dashboard_view_csv_table)
        btn_plots.clicked.connect(self.dashboard_view_plots)
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_table)
        btn_row.addWidget(btn_plots)
        btn_row.addStretch()

        # Image folder selectors
        btn_orig = self._btn("Original Images…", GREEN_MED)
        btn_pred = self._btn("Predicted Images…", BLUE_LIGHT)
        btn_orig.clicked.connect(lambda: self._set_dashboard_folder("orig"))
        btn_pred.clicked.connect(lambda: self._set_dashboard_folder("pred"))
        self._orig_folder_lbl = QLabel("")
        self._pred_folder_lbl = QLabel("")
        for lbl in (self._orig_folder_lbl, self._pred_folder_lbl):
            lbl.setMinimumWidth(120)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            lbl.setStyleSheet(f"background:{PATH_BG}; color:{TEXT_GREEN}; padding:2px 4px; border:1px solid {BORDER}; border-radius:3px;")
        btn_row.addWidget(btn_orig)
        btn_row.addWidget(self._orig_folder_lbl)
        btn_row.addWidget(btn_pred)
        btn_row.addWidget(self._pred_folder_lbl)
        layout.addLayout(btn_row)

        # ── Inner tabs: Results | Image Viewer ─────────────────────
        inner_tabs = QTabWidget()
        layout.addWidget(inner_tabs)

        results_page = QWidget()
        viewer_page  = QWidget()
        inner_tabs.addTab(results_page, "Results")
        inner_tabs.addTab(viewer_page,  "Image Viewer")

        # Results page placeholder
        results_lay = QVBoxLayout(results_page)
        lbl = QLabel(
            "Click 'Click to View Plots' to view interactive results."
        )
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            f"color:{BLUE_DARK}; font-size:13pt; font-weight:bold;"
        )
        results_lay.addWidget(lbl)

        # Image viewer page
        self._build_image_viewer(viewer_page)

    def _set_dashboard_folder(self, which):
        path = QFileDialog.getExistingDirectory(
            self, f"Select {'Original' if which == 'orig' else 'Predicted'} Images Folder"
        )
        if path:
            if which == "orig":
                self.orig_img_folder = path
                self._orig_folder_lbl.setText(Path(path).name)
            else:
                self.pred_img_folder = path
                self._pred_folder_lbl.setText(Path(path).name)
            self._refresh_image_viewer()

    def dashboard_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Analysis CSV", "", "CSV Files (*.csv)"
        )
        if path and Path(path).exists():
            self.last_analysis_csv = path

    def dashboard_view_csv_table(self):
        if not self.last_analysis_csv or not Path(self.last_analysis_csv).exists():
            QMessageBox.information(self, "No Data", "No analysis CSV loaded.")
            return
        try:
            import pandas as _pd
            df = _pd.read_csv(self.last_analysis_csv)
        except Exception as e:
            QMessageBox.critical(self, "CSV Error", f"Could not read CSV: {e}")
            return
        dlg = CSVTableDialog(self, df)
        dlg.exec_()

    def dashboard_view_plots(self):
        if not self.last_analysis_csv or not Path(self.last_analysis_csv).exists():
            QMessageBox.information(self, "No Data", "No analysis CSV loaded.")
            return
        try:
            self._dashboard_generate_and_open_plots(self.last_analysis_csv)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Error creating plots:\n{e}")

    # ── Image Viewer ───────────────────────────────────────────────
    def _build_image_viewer(self, parent):
        layout = QVBoxLayout(parent)
        layout.setSpacing(4)

        # Labels row
        label_row = QHBoxLayout()
        self._left_img_lbl  = QLabel("Original", alignment=Qt.AlignCenter)
        self._right_img_lbl = QLabel("Predicted", alignment=Qt.AlignCenter)
        for lbl in (self._left_img_lbl, self._right_img_lbl):
            lbl.setStyleSheet(f"color:{BLUE_DARK}; font-weight:bold;")
        label_row.addWidget(self._left_img_lbl)
        label_row.addWidget(self._right_img_lbl)
        layout.addLayout(label_row)

        # Image canvases
        canvas_row = QHBoxLayout()
        self._left_canvas  = ZoomPanLabel()
        self._right_canvas = ZoomPanLabel()
        canvas_row.addWidget(self._left_canvas)
        canvas_row.addWidget(self._right_canvas)
        layout.addLayout(canvas_row, stretch=1)

        # Navigation row
        nav_row = QHBoxLayout()
        self._btn_prev    = self._btn("⟨ Prev", BLUE_DARK, width=80)
        self._btn_next    = self._btn("Next ⟩", BLUE_DARK, width=80)
        self._img_counter = QLabel("")
        self._img_counter.setStyleSheet("font-size:10pt;")
        self._btn_prev.clicked.connect(self._viewer_prev)
        self._btn_next.clicked.connect(self._viewer_next)
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        nav_row.addWidget(self._img_counter)
        nav_row.addStretch()
        layout.addLayout(nav_row)

        # Thumbnail scroll bar
        thumb_scroll_area = QScrollArea()
        thumb_scroll_area.setFixedHeight(95)
        thumb_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        thumb_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        thumb_scroll_area.setWidgetResizable(True)
        self._thumb_widget = QWidget()
        self._thumb_layout = QHBoxLayout(self._thumb_widget)
        self._thumb_layout.setSpacing(2)
        self._thumb_layout.setContentsMargins(2, 2, 2, 2)
        self._thumb_layout.setAlignment(Qt.AlignLeft)
        thumb_scroll_area.setWidget(self._thumb_widget)
        thumb_scroll_area.setStyleSheet(f"background:{PATH_BG};")
        layout.addWidget(thumb_scroll_area)

        # Keyboard shortcuts
        for canvas in (self._left_canvas, self._right_canvas):
            canvas.setFocusPolicy(Qt.StrongFocus)
        self._left_canvas.keyPressEvent = self._viewer_key_press

        # Show placeholder
        self._show_viewer_placeholder()

    def _show_viewer_placeholder(self):
        self._left_canvas.setText(
            "Select both Original and Predicted image folders above!"
        )
        self._right_canvas.setText("No images")
        self._img_counter.setText("")

    def _refresh_image_viewer(self):
        import glob as _glob
        orig_dir = self.orig_img_folder
        pred_dir = self.pred_img_folder

        if not orig_dir or not Path(orig_dir).exists() \
                or not pred_dir or not Path(pred_dir).exists():
            self._show_viewer_placeholder()
            return

        orig_files = [p for p in Path(orig_dir).iterdir() if is_image(p)]
        pred_files = [p for p in Path(pred_dir).iterdir() if is_image(p)]
        orig_map   = {p.stem: p for p in orig_files}
        pred_map   = {p.stem: p for p in pred_files}
        matches    = sorted(set(orig_map.keys()) & set(pred_map.keys()))

        if not matches:
            self._left_canvas.setText("No matching basenames found between folders.")
            self._right_canvas.setText("No matches.")
            self._img_counter.setText("")
            return

        self._dashboard_matches  = matches
        self._dashboard_orig_map = {k: str(orig_map[k]) for k in matches}
        self._dashboard_pred_map = {k: str(pred_map[k]) for k in matches}
        self._dashboard_idx      = 0
        self._rebuild_thumbnails()
        self._viewer_show(0)

    def _rebuild_thumbnails(self):
        # Clear existing thumbnails
        while self._thumb_layout.count():
            item = self._thumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        from PIL import Image as _PILImage
        for i, base in enumerate(self._dashboard_matches):
            try:
                thumb = _PILImage.open(self._dashboard_orig_map[base]).resize((50, 50))
                data  = thumb.convert("RGBA").tobytes("raw", "RGBA")
                qimg  = QImage(data, 50, 50, QImage.Format_RGBA8888)
                pm    = QPixmap.fromImage(qimg)
            except Exception:
                pm = QPixmap(50, 50)
                pm.fill(QColor(BORDER))

            frame = QFrame()
            frame.setFixedSize(76, 80)
            frame.setStyleSheet(
                f"border:2px solid {'#21a179' if i == 0 else BORDER}; "
                f"border-radius:5px; "
                f"background:{'#e2ffe3' if i == 0 else '#f6f6f6'};"
            )
            f_lay = QVBoxLayout(frame)
            f_lay.setSpacing(1)
            f_lay.setContentsMargins(2, 2, 2, 2)

            img_lbl = QLabel()
            img_lbl.setPixmap(pm)
            img_lbl.setAlignment(Qt.AlignCenter)
            f_lay.addWidget(img_lbl)

            fname = base
            if len(fname) > 12:
                fname = fname[:11] + "…"
            txt_lbl = QLabel(fname)
            txt_lbl.setAlignment(Qt.AlignCenter)
            txt_lbl.setStyleSheet("font-size:7pt; border:none;")
            f_lay.addWidget(txt_lbl)

            # click handler
            frame.mousePressEvent = lambda e, idx=i: self._viewer_show(idx)
            img_lbl.mousePressEvent = lambda e, idx=i: self._viewer_show(idx)
            txt_lbl.mousePressEvent = lambda e, idx=i: self._viewer_show(idx)

            self._thumb_layout.addWidget(frame)

    def _viewer_show(self, idx):
        if not self._dashboard_matches:
            return
        self._dashboard_idx = idx
        base = self._dashboard_matches[idx]
        from PIL import Image as _PILImage
        try:
            img_o = _PILImage.open(self._dashboard_orig_map[base]).convert("RGB")
            img_p = _PILImage.open(self._dashboard_pred_map[base]).convert("RGB")
        except Exception as e:
            QMessageBox.warning(self, "Image Error", str(e))
            return
        self._left_canvas.set_pil_image(img_o)
        self._right_canvas.set_pil_image(img_p)
        self._left_img_lbl.setText(
            f"Original: {Path(self._dashboard_orig_map[base]).name}")
        self._right_img_lbl.setText(
            f"Predicted: {Path(self._dashboard_pred_map[base]).name}")
        self._img_counter.setText(
            f"Image {idx + 1} / {len(self._dashboard_matches)}")
        self._left_canvas.setFocus()

    def _viewer_prev(self):
        if self._dashboard_idx > 0:
            self._viewer_show(self._dashboard_idx - 1)

    def _viewer_next(self):
        if self._dashboard_idx < len(self._dashboard_matches) - 1:
            self._viewer_show(self._dashboard_idx + 1)

    def _viewer_key_press(self, event):
        if event.key() == Qt.Key_Left:
            self._viewer_prev()
        elif event.key() == Qt.Key_Right:
            self._viewer_next()
        else:
            QLabel.keyPressEvent(self._left_canvas, event)

    # ── Plots ──────────────────────────────────────────────────────
    def _dashboard_generate_and_open_plots(self, csv_path):
        import pandas as _pd
        import plotly.graph_objs as go
        import plotly.express as px
        import numpy as np
        from scipy.cluster.hierarchy import linkage, leaves_list

        df = _pd.read_csv(csv_path)
        html_blocks = []
        classes = [c for c in df.columns if c not in ("Image", "Group")]
        class_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Vivid

        # 1. Absolute cell counts
        if "Group" in df.columns:
            group_abs = df.groupby("Group")[classes].sum()
            fig1 = go.Figure()
            for i, group in enumerate(group_abs.index):
                fig1.add_trace(go.Bar(
                    x=classes, y=group_abs.loc[group].values,
                    name=str(group),
                    marker_color=class_palette[i % len(class_palette)],
                    text=[str(int(v)) for v in group_abs.loc[group].values],
                    textposition="outside",
                    hovertemplate=f"Group: {group}<br>Class: %{{x}}<br>Count: %{{y}}<extra></extra>"
                ))
            fig1.update_layout(barmode="group",
                               title="Absolute Cell Counts by Class per Group",
                               xaxis_title="Class", yaxis_title="Total Count",
                               template="plotly_white")
        else:
            abs_sum = df[classes].sum()
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(x=classes, y=abs_sum,
                                  marker_color=class_palette[:len(classes)],
                                  text=[str(int(v)) for v in abs_sum],
                                  textposition="outside"))
            fig1.update_layout(title="Absolute Cell Counts by Class",
                               xaxis_title="Class", yaxis_title="Total Count",
                               template="plotly_white")
        html_blocks.append(fig1.to_html(include_plotlyjs="cdn", full_html=False))

        # Relative abundance
        if "Group" in df.columns:
            group_totals = df.groupby("Group")[classes].sum()
            group_total_counts = group_totals.sum(axis=1)
            fig_rel = go.Figure()
            for i, group in enumerate(group_totals.index):
                rel_pct = (group_totals.loc[group] / group_total_counts[group] * 100).round(2)
                fig_rel.add_trace(go.Bar(
                    x=classes, y=rel_pct, name=str(group),
                    marker_color=class_palette[i % len(class_palette)],
                    text=[f"{v:.1f}%" for v in rel_pct],
                    textposition="outside",
                    hovertemplate=f"Group: {group}<br>Class: %{{x}}<br>Rel %: %{{y:.2f}}%<extra></extra>"
                ))
            fig_rel.update_layout(barmode="group",
                                  title="Relative Abundance (%) by Class per Group",
                                  xaxis_title="Class", yaxis_title="(%) Abundance",
                                  template="plotly_white")
        else:
            total = df[classes].sum()
            total_cells = total.sum()
            rel_pct = (total / total_cells * 100).round(2)
            fig_rel = go.Figure(data=[go.Bar(
                x=classes, y=rel_pct,
                text=[f"{v:.1f}%" for v in rel_pct],
                textposition="outside",
                marker_color=class_palette[:len(classes)]
            )])
            fig_rel.update_layout(title="Relative Abundance (%) of Morphotypes",
                                  xaxis_title="Class", yaxis_title="(%) Abundance",
                                  template="plotly_white")
        html_blocks.append(fig_rel.to_html(include_plotlyjs=False, full_html=False))

        # Mean relative abundance
        rel_df = df[classes].div(df[classes].sum(axis=1).replace(0, 1), axis=0)
        if "Group" in df.columns:
            rel_mean = rel_df.copy()
            rel_mean["Group"] = df["Group"]
            rel_grp = rel_mean.groupby("Group").mean() * 100
            fig2 = go.Figure()
            for i, group in enumerate(rel_grp.index):
                fig2.add_trace(go.Bar(
                    x=classes, y=rel_grp.loc[group].values,
                    name=str(group),
                    marker_color=class_palette[i % len(class_palette)],
                    text=[f"{v:.1f}%" for v in rel_grp.loc[group].values],
                    textposition="outside",
                    hovertemplate=f"Group: {group}<br>Class: %{{x}}<br>Mean %: %{{y:.2f}}%<extra></extra>"
                ))
            fig2.update_layout(barmode="group",
                               title="Mean Relative Abundance (%) by Class per Group",
                               xaxis_title="Class",
                               yaxis_title="Mean Relative Abundance (%)",
                               template="plotly_white")
        else:
            rel_mean_vals = rel_df.mean() * 100
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=classes, y=rel_mean_vals,
                                  marker_color=class_palette[:len(classes)],
                                  text=[f"{v:.1f}%" for v in rel_mean_vals],
                                  textposition="outside"))
            fig2.update_layout(title="Mean Relative Abundance (%) by Class",
                               xaxis_title="Class",
                               yaxis_title="Mean Relative Abundance (%)",
                               template="plotly_white")
        html_blocks.append(fig2.to_html(include_plotlyjs=False, full_html=False))

        # Normalized stacked bar
        if "Group" in df.columns:
            prop_df = df.groupby("Group")[classes].sum()
            prop_df = prop_df.div(prop_df.sum(axis=1), axis=0).fillna(0)
            x = prop_df.index
        else:
            prop_df = rel_df
            x = df["Image"]
        fig3 = go.Figure()
        for i, cls in enumerate(classes):
            fig3.add_trace(go.Bar(
                x=x, y=prop_df[cls] * 100, name=cls,
                marker_color=class_palette[i % len(class_palette)],
                hovertemplate=f"%{{y:.2f}}% {cls}<extra></extra>"
            ))
        fig3.update_layout(barmode="stack",
                           title="Normalized Stacked Bar Plot (Class Composition)",
                           xaxis_title="Image/Group", yaxis_title="%",
                           template="plotly_white")
        html_blocks.append(fig3.to_html(include_plotlyjs=False, full_html=False))

        # Heatmaps
        colorscale1, colorscale2 = "Viridis", "Plasma"
        if "Group" not in df.columns:
            rel_img = df[classes].div(df[classes].sum(axis=1).replace(0, 1), axis=0)
            col_labels = df["Image"].astype(str).tolist()
            if len(rel_img) > 1:
                Z_col = linkage(rel_img.values, method="ward", metric="euclidean")
                col_leaves = leaves_list(Z_col)
                col_labels = [col_labels[i] for i in col_leaves]
                rel_img = rel_img.iloc[col_leaves, :]
            fig4 = go.Figure(data=go.Heatmap(
                z=rel_img.values.T, x=col_labels, y=rel_img.columns,
                colorscale=colorscale2, colorbar=dict(title="Rel. Abundance"),
                zmin=0, zmax=1
            ))
            fig4.update_layout(
                title="Heatmap of Morphotype Distribution per Image (Relative Abundance, Clustered)",
                xaxis_title="Image (clustered)", yaxis_title="Class",
                template="plotly_white"
            )
            html_blocks.append(fig4.to_html(include_plotlyjs=False, full_html=False))

        if "Group" in df.columns:
            grp_prop = df.groupby("Group")[classes].mean()
            grp_rel  = grp_prop.div(grp_prop.sum(axis=1), axis=0).fillna(0)
            if grp_rel.shape[0] > 1:
                Z_col = linkage(grp_rel.values, method="ward", metric="euclidean")
                col_leaves = leaves_list(Z_col)
                grp_names = list(grp_rel.index[col_leaves])
                grp_rel   = grp_rel.iloc[col_leaves, :]
            else:
                grp_names = list(grp_rel.index)
            fig5 = go.Figure(data=go.Heatmap(
                z=grp_rel.values.T, x=grp_names, y=grp_rel.columns,
                colorscale=colorscale1, colorbar=dict(title="Rel. Abundance"),
                zmin=0, zmax=1
            ))
            fig5.update_layout(
                title="Heatmap of Group-wise Morphotype Distribution (Clustered)",
                xaxis_title="Group (clustered)", yaxis_title="Class",
                template="plotly_white"
            )
            html_blocks.append(fig5.to_html(include_plotlyjs=False, full_html=False))

            rel_img = df[classes].div(df[classes].sum(axis=1).replace(0, 1), axis=0).fillna(0)
            col_labels = [f"{g}_{i}" for g, i in zip(df["Group"], df["Image"])]
            if len(rel_img) > 1:
                Z_col = linkage(rel_img.values, method="ward", metric="euclidean")
                col_leaves = leaves_list(Z_col)
                col_labels = [col_labels[i] for i in col_leaves]
                rel_img    = rel_img.iloc[col_leaves, :]
            fig6 = go.Figure(data=go.Heatmap(
                z=rel_img.values.T, x=col_labels, y=rel_img.columns,
                colorscale=colorscale2, colorbar=dict(title="Rel. Abundance"),
                zmin=0, zmax=1
            ))
            fig6.update_layout(
                title="Heatmap of Morphotype Distribution per Image (Clustered)",
                xaxis_title="Image (clustered)", yaxis_title="Class",
                template="plotly_white"
            )
            html_blocks.append(fig6.to_html(include_plotlyjs=False, full_html=False))

        html = (
            "<html><head><meta charset='utf-8'></head><body>"
            + "\n<hr>\n".join(html_blocks)
            + "</body></html>"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html",
                                         encoding="utf-8") as tmp:
            tmp.write(html)
            html_path = tmp.name
        webbrowser.open(Path(html_path).as_uri())

    # ──────────────────────────────────────────────────────────────
    # Utility widget factories
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _btn(text, bg_color, text_color="white", font_size=12,
             height=None, width=None):
        btn = QPushButton(text)
        # Scale font_size relative to the application's base font point size
        app_pt = QApplication.font().pointSize()
        if app_pt <= 0:
            app_pt = 10
        # font_size arg is treated as a relative scale in px at 96 DPI base;
        # convert to pt and scale proportionally
        scaled_pt = max(7, round(app_pt * font_size / 10))
        style = (
            f"background-color:{bg_color}; color:{text_color}; "
            f"font-size:{scaled_pt}pt; font-weight:bold; "
            f"border-radius:6px; padding:0.4em 0.9em; border:none;"
        )
        btn.setStyleSheet(style)
        # Use expanding size policy so buttons grow/shrink with the window
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        if height:
            btn.setMinimumHeight(height)
            btn.setMaximumHeight(int(height * 1.5))
        if width:
            btn.setMinimumWidth(width)
            btn.setMaximumWidth(int(width * 2))
        return btn

    @staticmethod
    def _path_label(text=""):
        lbl = QLabel(text)
        lbl.setObjectName("path_label")
        lbl.setWordWrap(True)
        lbl.setMinimumHeight(24)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return lbl


# ═══════════════════════════════════════════════════════════════════
# Import Thread (splash loader)
# ═══════════════════════════════════════════════════════════════════
class _ImportThread(QThread):
    done = pyqtSignal()

    def run(self):
        global pd, np, cv2
        try:
            import pandas as _pd;  pd  = _pd
            import numpy  as _np;  np  = _np
            import cv2    as _cv2; cv2 = _cv2
            import scipy
            import plotly.graph_objs
        except ImportError:
            pass
        self.done.emit()


# ═══════════════════════════════════════════════════════════════════
# Prediction Worker Thread
# ═══════════════════════════════════════════════════════════════════
class PredictionThread(WorkerThread):
    def __init__(self, input_folder, output_folder, confidence, line_width,
                 selected_cls, agnostic_nms, show_output, analysis_name,
                 save_images, custom_names_enabled, custom_class_names):
        super().__init__()
        self.input_folder        = input_folder
        self.output_folder       = output_folder
        self.confidence          = confidence
        self.line_width          = line_width
        self.selected_cls        = selected_cls
        self.agnostic_nms        = agnostic_nms
        self.show_output         = show_output
        self.analysis_name       = analysis_name
        self.save_images         = save_images
        self.custom_names_enabled = custom_names_enabled
        self.custom_class_names  = custom_class_names

    def run(self):
        try:
            from ultralytics import YOLO
            import cv2 as _cv2

            model_path = resource_path("TU_MyCo-Vision.pt")
            model = YOLO(str(model_path))

            img_folder = Path(self.input_folder)
            # Determine imgsz from first valid image
            imgsz = (640, 640)
            for f in img_folder.iterdir():
                if is_image(f):
                    img = _cv2.imread(str(f))
                    if img is not None:
                        h, w = img.shape[:2]
                        imgsz = (max(32, round(h / 32) * 32),
                                 max(32, round(w / 32) * 32))
                        break

            self.progress.emit(0.1)
            model.predict(
                source=str(img_folder),
                imgsz=imgsz,
                conf=self.confidence,
                line_width=self.line_width,
                classes=self.selected_cls,
                save=self.save_images,
                save_txt=True,
                save_conf=True,
                agnostic_nms=self.agnostic_nms,
                augment=True,
                show=self.show_output,
                name=self.analysis_name,
                project=self.output_folder,
                max_det=1000,
            )
            self.progress.emit(0.8)

            # If custom names enabled, draw overlays from label files
            if self.custom_names_enabled:
                pred_folder = Path(self.output_folder) / self.analysis_name
                label_dir   = pred_folder / "labels"
                if not label_dir.exists():
                    label_dir = pred_folder
                overlay_dir = pred_folder / "custom_overlays"
                # custom_class_names is in canonical (CLASSES_ALL) order;
                # draw_yolo_overlays indexes by YOLO class ID → convert first
                yolo_ordered_names = _canonical_to_yolo_names(self.custom_class_names)
                written = draw_yolo_overlays(
                    image_dir   = img_folder,
                    label_dir   = label_dir,
                    output_dir  = overlay_dir,
                    custom_names= yolo_ordered_names,
                )
                self.progress.emit(1.0)
                self.finished.emit(
                    f"YOLO Prediction completed!\n"
                    f"Custom overlays ({written} images) saved to:\n"
                    f"{overlay_dir}"
                )
            else:
                self.progress.emit(1.0)
                self.finished.emit("YOLO Prediction completed!")

        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════
# Analysis helper — shared label parsing logic
# ═══════════════════════════════════════════════════════════════════

# Canonical display order of the 13 classes (used for CSV columns / plots)
CLASSES_ALL    = ["C1", "C2", "C2-B", "C3-B", "C5", "C8", "C9",
                  "C4", "C6", "C11", "C7-E", "C7-F", "C7-G"]

# YOLO model class-ID order (id 0–12).
# draw_yolo_overlays and CustomClassNamesDialog use this ordering.
CLASSES_YOLO_ORDER = ["C2", "C5", "C1", "C3-B", "C4", "C11", "C9",
                      "C2-B", "C8", "C6", "C7-E", "C7-F", "C7-G"]

# Maps canonical index → YOLO id  (i.e. CLASSES_ALL[i] has YOLO id CORRECT_ALL[i])
CORRECT_ALL    = [2, 0, 7, 8, 1, 9, 6, 3, 4, 5, 10, 11, 12]
EPSILON        = 0.001


def _canonical_to_yolo_names(canonical_names: list) -> list:
    """
    Given a list of 13 custom names in CANONICAL order (matching CLASSES_ALL),
    return a list of 13 names in YOLO class-ID order (matching CLASSES_YOLO_ORDER).
    This is the order required by draw_yolo_overlays.
    """
    # Build: yolo_id → custom name
    # CORRECT_ALL[i] is the YOLO id for canonical index i
    yolo_names = [""] * 13
    for canonical_idx, yolo_id in enumerate(CORRECT_ALL):
        yolo_names[yolo_id] = canonical_names[canonical_idx]
    return yolo_names


def _yolo_to_canonical_names(yolo_names: list) -> list:
    """
    Given a list of 13 custom names in YOLO class-ID order (matching CLASSES_YOLO_ORDER),
    return a list of 13 names in CANONICAL order (matching CLASSES_ALL).
    This is the order required for CSV columns and plot labels.
    """
    # CORRECT_ALL[i] = yolo_id for canonical index i
    canonical_names = [""] * 13
    for canonical_idx, yolo_id in enumerate(CORRECT_ALL):
        canonical_names[canonical_idx] = yolo_names[yolo_id]
    return canonical_names


def _parse_label_file(path: Path) -> list:
    """Parse one YOLO label .txt file and return a list of 13 counts."""
    counts = [0] * 13
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            cid = int(vals[0])
            cx, cy, bw, bh = vals[1], vals[2], vals[3], vals[4]
            marginx = (
                abs((cx + bw / 2) - 1) < EPSILON or
                (cx - bw / 2) < EPSILON
            )
            marginy = (
                abs((cy + bh / 2) - 1) < EPSILON or
                (cy - bh / 2) < EPSILON
            )
            if (marginx or marginy) and cid not in [4, 10, 11, 12]:
                continue
            if 0 <= cid < 13:
                counts[cid] += 1
    # reorder from YOLO ids to canonical class order
    ordered = [counts[j] for j in CORRECT_ALL]
    return ordered


def _apply_custom_names_to_df(df, custom_names):
    """Rename class columns in df using custom_names if provided."""
    if not custom_names:
        return df
    rename_map = {old: new for old, new in zip(CLASSES_ALL, custom_names)}
    return df.rename(columns=rename_map)


# ═══════════════════════════════════════════════════════════════════
# Single Group Analysis Thread
# ═══════════════════════════════════════════════════════════════════
class SingleGroupAnalysisThread(WorkerThread):
    csv_ready = pyqtSignal(str)

    def __init__(self, folder, analysis_name, save_dir,
                 excluded_indices, custom_names=None):
        super().__init__()
        self.folder           = folder
        self.analysis_name    = analysis_name
        self.save_dir         = save_dir
        self.excluded_indices = excluded_indices
        self.custom_names     = custom_names  # None or list of 13 strings

    def run(self):
        try:
            import pandas as _pd
            self.progress.emit(0.1)

            folder_path = Path(self.folder)
            label_dir   = folder_path / "labels"
            if not label_dir.exists():
                label_dir = folder_path
            label_files = sorted(label_dir.glob("*.txt"))
            if not label_files:
                self.error.emit("No label .txt files found in selected folder.")
                return

            image_names, all_counts_raw = [], []
            for idx, path in enumerate(label_files):
                ordered = _parse_label_file(path)
                all_counts_raw.append(ordered)
                image_names.append(path.stem)
                self.progress.emit(0.1 + 0.6 * (idx + 1) / len(label_files))

            save_path = Path(self.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            df_raw = _pd.DataFrame(all_counts_raw, columns=CLASSES_ALL)
            df_raw.insert(0, "Image", image_names)

            # Apply custom names before saving
            df_out = _apply_custom_names_to_df(df_raw, self.custom_names)
            csv_path = str(save_path / f"{self.analysis_name}_rawdata.csv")
            df_out.to_csv(csv_path, index=False, encoding="utf-8")
            self.csv_ready.emit(csv_path)

            # Exclusion
            if self.excluded_indices:
                # determine column names to exclude
                excl_cols = [CLASSES_ALL[i] for i in self.excluded_indices]
                if self.custom_names:
                    excl_cols = [self.custom_names[i] for i in self.excluded_indices]
                all_data_cols = [c for c in df_out.columns if c != "Image"]
                df_mod = df_out.copy()
                present_excl = [c for c in excl_cols if c in df_mod.columns]
                if present_excl:
                    df_mod["Others"] = df_mod[present_excl].sum(axis=1)
                    df_mod.drop(columns=present_excl, inplace=True)
                df_mod.to_csv(
                    save_path / f"{self.analysis_name}_excluded.csv",
                    index=False, encoding="utf-8"
                )

            self.progress.emit(1.0)
            self.finished.emit(
                f"Single Group CSVs saved in:\n{save_path}"
            )
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════
# Multi Group Analysis Thread
# ═══════════════════════════════════════════════════════════════════
class MultiGroupAnalysisThread(WorkerThread):
    csv_ready = pyqtSignal(str)

    def __init__(self, folders, analysis_name, save_dir,
                 excluded_indices, custom_names=None):
        super().__init__()
        self.folders          = folders
        self.analysis_name    = analysis_name
        self.save_dir         = save_dir
        self.excluded_indices = excluded_indices
        self.custom_names     = custom_names

    def run(self):
        try:
            import pandas as _pd
            self.progress.emit(0.1)

            all_counts, group_labels, image_labels = [], [], []
            for g_idx, folder in enumerate(self.folders):
                folder_path = Path(folder)
                label_dir   = folder_path / "labels"
                if not label_dir.exists():
                    label_dir = folder_path
                files = sorted(label_dir.glob("*.txt"))
                for path in files:
                    ordered = _parse_label_file(path)
                    all_counts.append(ordered)
                    group_labels.append(folder_path.name)
                    image_labels.append(path.stem)
                self.progress.emit(
                    0.1 + 0.8 * (g_idx + 1) / len(self.folders))

            if not all_counts:
                self.error.emit("No label .txt files found in any group folder.")
                return

            save_path = Path(self.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            df_all = _pd.DataFrame(all_counts, columns=CLASSES_ALL)
            df_all["Group"] = group_labels
            df_all["Image"] = image_labels

            df_out = _apply_custom_names_to_df(df_all, self.custom_names)
            csv_path = str(save_path / f"{self.analysis_name}_rawdata.csv")
            df_out.to_csv(csv_path, index=False, encoding="utf-8")
            self.csv_ready.emit(csv_path)

            if self.excluded_indices:
                excl_cols = [CLASSES_ALL[i] for i in self.excluded_indices]
                if self.custom_names:
                    excl_cols = [self.custom_names[i] for i in self.excluded_indices]
                df_mod = df_out.copy()
                present_excl = [c for c in excl_cols if c in df_mod.columns]
                if present_excl:
                    df_mod["Others"] = df_mod[present_excl].sum(axis=1)
                    df_mod.drop(columns=present_excl, inplace=True)
                df_mod.to_csv(
                    save_path / f"{self.analysis_name}_excluded.csv",
                    index=False, encoding="utf-8"
                )

            self.progress.emit(1.0)
            self.finished.emit(
                f"Multi-Group CSVs saved in:\n{save_path}"
            )
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # High-DPI support
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Ensure font renders crisply — use pt size so it scales with system DPI
    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)

    window = MycoVisionApp()
    window.show()
    sys.exit(app.exec_())
