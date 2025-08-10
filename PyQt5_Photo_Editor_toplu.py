"""
PyQt5 Photo Editor with Tabs — single-file app
Integrated from user's JS-like features + previous single-image PyQt app.
Each tab contains one ImageDocument (pil image + history + adjustments).
Controls on the right operate on the currently active tab.
"""

import sys
import os
import math
from functools import partial
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import cv2

from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QByteArray
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QGridLayout, QSlider, QSplitter, QSizePolicy,
    QMessageBox, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem,
    QListWidget, QListWidgetItem, QTabWidget, QAction
)

# Matplotlib for histogram rendering
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------- Utilities ----------------------
def pil_to_qpixmap(im: Image.Image) -> QPixmap:
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    data = im.tobytes('raw', 'RGBA')
    qimg = QImage(data, im.width, im.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)

def kelvin_to_rgb_gains(kelvin: int):
    temp = kelvin / 100.0
    def clamp(v, lo=0, hi=255):
        return max(lo, min(hi, v))
    if temp <= 66:
        r = 255
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592)
        r = clamp(r)
    if temp <= 66:
        g = 99.4708025861 * math.log(temp) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
    g = clamp(g)
    if temp >= 66:
        b = 255
    elif temp <= 19:
        b = 0
    else:
        b = 138.5177312231 * math.log(temp - 10) - 305.0447927307
        b = clamp(b)
    return (r/255.0, g/255.0, b/255.0)

def compute_histogram(pil_img: Image.Image):
    im = pil_img.convert('RGB')
    arr = np.array(im)
    r = arr[:,:,0].ravel()
    g = arr[:,:,1].ravel()
    b = arr[:,:,2].ravel()
    lum = (0.2126*r + 0.7152*g + 0.0722*b).astype(np.uint8).ravel()
    bins = 256
    hr = np.bincount(r, minlength=bins)
    hg = np.bincount(g, minlength=bins)
    hb = np.bincount(b, minlength=bins)
    hl = np.bincount(lum, minlength=bins)
    return {'r': hr, 'g': hg, 'b': hb, 'lum': hl, 'total': im.width*im.height}

def histogram_pixmap(hist, w=400, h=120):
    fig = Figure(figsize=(w/100.0, h/100.0), dpi=100, tight_layout=True)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(hist['lum'], color='k', linewidth=1.2)
    ax.plot(hist['r'], color='#c00', linewidth=0.8)
    ax.plot(hist['g'], color='#0a0', linewidth=0.8)
    ax.plot(hist['b'], color='#00c', linewidth=0.8)
    ax.axis('off')
    canvas.draw()
    buf = canvas.buffer_rgba()
    w_px, h_px = int(fig.get_size_inches()[0]*fig.get_dpi()), int(fig.get_size_inches()[1]*fig.get_dpi())
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h_px, w_px, 4))
    img = Image.fromarray(arr)
    return pil_to_qpixmap(img)

# ---------------------- Data Model ----------------------
class ImageDocument:
    def __init__(self, path=None, pil_image: Image.Image=None):
        self.path = path
        self.pil = pil_image.convert('RGB') if pil_image else None
        self.history = []
        if self.pil:
            self.history.append(self.pil.copy())
        self.adjustments = {
            'brightness': 0,    # -100..100
            'contrast': 0,      # -100..100
            'saturation': 0,    # -100..100
            'kelvin': 6500,     # 2000..10000
            'shadows': 0,       # -100..100
            'highlights': 0     # -100..100
        }
    def push(self):
        self.history.append(self.pil.copy())
    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.pil = self.history[-1].copy()
            return True
        return False
    def reset_adjustments(self):
        self.adjustments = {k: (6500 if k=='kelvin' else 0) for k in self.adjustments}

# ---------------------- Tab widget (holds view + scene + reference to doc) ----------------------
class PhotoTab(QWidget):
    def __init__(self, doc: ImageDocument):
        super().__init__()
        self.doc = doc
        layout = QVBoxLayout(self)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        # rendering hints
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        layout.addWidget(self.view)
        self.pix_item = None
        self.refresh()

    def refresh(self):
        self.scene.clear()
        if self.doc and self.doc.pil:
            pix = pil_to_qpixmap(self.doc.pil)
            self.pix_item = QGraphicsPixmapItem(pix)
            self.scene.addItem(self.pix_item)
            self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)

# ---------------------- Main Window ----------------------
class PhotoEditorMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Photo Editor — Tabs")
        self.resize(1200, 800)

        # central widget with splitter: left list, right split (tabs + controls)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left: thumbnail/list of opened images (optional quick select)
        self.list_widget = QListWidget()
        self.list_widget.setMaximumWidth(240)
        self.list_widget.itemClicked.connect(self.on_list_click)
        main_layout.addWidget(self.list_widget)

        # Middle-right: tabs + control panel
        right_split = QSplitter(Qt.Horizontal)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        right_split.addWidget(self.tabs)

        # Control panel (shared, works on current tab)
        control_panel = QWidget()
        cp_layout = QVBoxLayout(control_panel)
        cp_layout.setSpacing(8)

        cp_layout.addWidget(QLabel("Histogram (Luma + RGB)"))
        self.hist_label = QLabel()
        self.hist_label.setFixedHeight(100)
        self.hist_label.setStyleSheet('border:1px solid #ddd; background:#fff')
        cp_layout.addWidget(self.hist_label)

        # sliders
        grid = QGridLayout()
        grid.setSpacing(6)
        self.sliders = {}
        def add_slider(title, minv, maxv, init, row):
            lab = QLabel(title)
            s = QSlider(Qt.Horizontal)
            s.setRange(minv, maxv); s.setValue(init)
            val_label = QLabel(str(init))
            s.valueChanged.connect(lambda v, t=title: self.on_slider(t, v))
            s.valueChanged.connect(lambda v, l=val_label: l.setText(str(v)))
            self.sliders[title] = s
            grid.addWidget(lab, row, 0)
            grid.addWidget(s, row, 1)
            grid.addWidget(val_label, row, 2)
        add_slider('Parlaklık', -100, 100, 0, 0)
        add_slider('Kontrast', -100, 100, 0, 1)
        add_slider('Doygunluk', -100, 100, 0, 2)
        add_slider('Beyaz Dengesi (K)', 2000, 10000, 6500, 3)
        add_slider('Shadows', -100, 100, 0, 4)
        add_slider('Highlights', -100, 100, 0, 5)

        cp_layout.addLayout(grid)

        # Buttons row (effects)
        def btn(text, cb): 
            b = QPushButton(text); b.clicked.connect(cb); return b
        row1 = QHBoxLayout()
        row1.addWidget(btn('90° Döndür', self.rotate90))
        row1.addWidget(btn('Yatay Çevir', self.flip_horizontal))
        row1.addWidget(btn('Geri Al', self.undo))
        cp_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(btn('Keskinleştir', partial(self.apply_effect, 'sharpen')))
        row2.addWidget(btn('Turunculaştır', partial(self.apply_effect, 'orange')))
        row2.addWidget(btn('Kırmızılık', partial(self.apply_effect, 'red')))
        row2.addWidget(btn('Mavi Ton', partial(self.apply_effect, 'blue')))
        cp_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(btn('Beyazlat', partial(self.apply_effect, 'brighten')))
        row3.addWidget(btn('Clarity', partial(self.apply_effect, 'clarity')))
        row3.addWidget(btn('Vignette', partial(self.apply_effect, 'vignette')))
        row3.addWidget(btn('Noise Red.', partial(self.apply_effect, 'noise')))
        cp_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(btn('Röportaj Modu', self.portrait_mode))
        row4.addWidget(btn('Otomatik İyileştir', self.auto_enhance))
        row4.addWidget(btn('Sıfırla', self.reset_adjustments))
        row4.addWidget(btn('İndir', self.export_current))
        cp_layout.addLayout(row4)

        # Bulk export and filler
        cp_layout.addStretch(1)

        right_split.addWidget(control_panel)
        right_split.setStretchFactor(0, 4)
        right_split.setStretchFactor(1, 1)

        main_layout.addWidget(right_split)

        # Menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Dosya")
        open_action = QAction("Görsel Aç...", self); open_action.triggered.connect(self.load_images)
        file_menu.addAction(open_action)
        bulk_action = QAction("Toplu Dışa Aktar...", self); bulk_action.triggered.connect(self.bulk_export)
        file_menu.addAction(bulk_action)
        exit_action = QAction("Çıkış", self); exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.tabs_docs = []  # parallel to tabs: PhotoTab instances

    # ---------------------- Tab & Loading ----------------------
    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Resimleri Seç', os.getcwd(), 'Images (*.png *.jpg *.jpeg *.bmp *.webp)')
        if not paths: return
        for p in paths:
            try:
                im = Image.open(p).convert('RGB')
            except Exception as e:
                QMessageBox.warning(self, 'Hata', f'"{p}" açılamadı: {e}')
                continue
            doc = ImageDocument(path=p, pil_image=im)
            self.add_tab(doc)

    def add_tab(self, doc: ImageDocument):
        tab = PhotoTab(doc)
        name = os.path.basename(doc.path) if doc.path else f'Resim {len(self.tabs_docs)+1}'
        index = self.tabs.addTab(tab, name)
        self.tabs.setCurrentIndex(index)
        self.tabs_docs.append(tab)
        # add to left list for quick select
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, tab)
        self.list_widget.addItem(item)
        self.list_widget.setCurrentItem(item)
        self.refresh_controls_for_current()

    def close_tab(self, index):
        tab = self.tabs.widget(index)
        reply = QMessageBox.question(self, "Sekmeyi Kapat", f"\"{self.tabs.tabText(index)}\" sekmesini kapatmak istiyor musunuz?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # remove from left list
            # find associated QListWidgetItem
            for i in range(self.list_widget.count()):
                it = self.list_widget.item(i)
                if it.data(Qt.UserRole) is tab:
                    self.list_widget.takeItem(i)
                    break
            self.tabs.removeTab(index)
            self.tabs_docs.remove(tab)
            self.refresh_controls_for_current()

    def on_tab_changed(self, index):
        self.refresh_controls_for_current()

    def on_list_click(self, item: QListWidgetItem):
        tab = item.data(Qt.UserRole)
        idx = self.tabs.indexOf(tab)
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)

    def current_tab(self) -> PhotoTab:
        w = self.tabs.currentWidget()
        return w

    def current_doc(self) -> ImageDocument:
        t = self.current_tab()
        return t.doc if t else None

    # ---------------------- Controls sync ----------------------
    def refresh_controls_for_current(self):
        doc = self.current_doc()
        if not doc:
            # disable sliders
            for s in self.sliders.values():
                s.blockSignals(True); s.setValue(0); s.blockSignals(False)
            self.hist_label.clear()
            return
        # set sliders to doc.adjustments (block signals to avoid immediate preview)
        mapping = {
            'Parlaklık': 'brightness', 'Kontrast': 'contrast', 'Doygunluk': 'saturation',
            'Beyaz Dengesi (K)': 'kelvin', 'Shadows': 'shadows', 'Highlights': 'highlights'
        }
        for slider_name, key in mapping.items():
            s = self.sliders[slider_name]
            s.blockSignals(True)
            s.setValue(doc.adjustments[key])
            s.blockSignals(False)
        # update view & histogram
        t = self.current_tab()
        if t:
            t.refresh()
            hist = compute_histogram(doc.pil)
            pix = histogram_pixmap(hist, w=400, h=100)
            self.hist_label.setPixmap(pix.scaled(self.hist_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_slider(self, name, value):
        doc = self.current_doc()
        if not doc:
            return
        mapping = {
            'Parlaklık': 'brightness', 'Kontrast': 'contrast', 'Doygunluk': 'saturation',
            'Beyaz Dengesi (K)': 'kelvin', 'Shadows': 'shadows', 'Highlights': 'highlights'
        }
        if name not in mapping:
            return
        key = mapping[name]
        doc.adjustments[key] = value
        self.apply_adjustments_preview()

    def apply_adjustments_preview(self):
        doc = self.current_doc()
        if not doc:
            return
        base = doc.history[-1]
        ad = doc.adjustments
        arr = np.array(base).astype(np.float32)
        # white balance
        gains = kelvin_to_rgb_gains(ad['kelvin'])
        arr[:,:,0] = np.clip(arr[:,:,0] * gains[0], 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1] * gains[1], 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2] * gains[2], 0, 255)
        # brightness
        arr = np.clip(arr + ad['brightness'], 0, 255)
        # contrast
        c = 1 + (ad['contrast'] / 100.0)
        arr = np.clip((arr - 128) * c + 128, 0, 255)
        # saturation
        if ad['saturation'] != 0:
            s = ad['saturation'] / 100.0
            lum = (0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2])[:,:,None]
            arr = np.clip(lum + (arr - lum) * (1 + s), 0, 255)
        # shadows/highlights approx
        def tone_pixel(a, shadows, highlights):
            t = a/255.0
            s = shadows/100.0
            h = highlights/100.0
            if s != 0:
                lift = s*0.6
                w = np.minimum(1.0, np.maximum(0.0, (t - 0.0)/(0.6-0.0)))
                t = t + (lift*(1-w))*(1-t)
            if h != 0:
                comp = h*0.6
                w2 = np.minimum(1.0, np.maximum(0.0, (t - 0.4)/(1.0-0.4)))
                t = t - (comp * w2) * t
            return np.clip(t*255.0, 0, 255)
        if ad['shadows'] != 0 or ad['highlights'] != 0:
            arr = tone_pixel(arr, ad['shadows'], ad['highlights'])
        img2 = Image.fromarray(arr.astype(np.uint8))
        # set doc.pil to preview but do not push history
        doc.pil = img2
        # refresh current tab view & histogram
        t = self.current_tab()
        if t:
            t.refresh()
            hist = compute_histogram(doc.pil)
            pix = histogram_pixmap(hist, w=400, h=100)
            self.hist_label.setPixmap(pix.scaled(self.hist_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ---------------------- Effects (applied and pushed) ----------------------
    def apply_effect(self, effect):
        doc = self.current_doc()
        if not doc:
            return
        img = doc.pil.copy()
        if effect == 'sharpen':
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        elif effect == 'orange':
            r, g, b = img.split()
            r = r.point(lambda v: min(255, v+12))
            g = g.point(lambda v: min(255, v+6))
            img = Image.merge('RGB', (r,g,b))
        elif effect == 'red':
            r, g, b = img.split()
            r = r.point(lambda v: min(255, v+18))
            img = Image.merge('RGB', (r,g,b))
        elif effect == 'blue':
            r, g, b = img.split()
            b = b.point(lambda v: min(255, v+18))
            img = Image.merge('RGB', (r,g,b))
        elif effect == 'brighten':
            img = ImageEnhance.Brightness(img).enhance(1.15)
        elif effect == 'clarity':
            img = ImageEnhance.Sharpness(img).enhance(1.2)
        elif effect == 'vignette':
            w, h = img.size
            xv, yv = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
            dist = np.sqrt(xv**2 + yv**2)
            mask = np.clip(1 - (dist/np.sqrt(2)), 0, 1)
            mask = (0.6 + 0.4*mask)[:,:,None]
            arr = np.array(img).astype(np.float32)
            arr = arr * mask
            img = Image.fromarray(arr.astype(np.uint8))
        elif effect == 'noise':
            a = np.array(img)
            a = cv2.bilateralFilter(a, d=5, sigmaColor=75, sigmaSpace=75)
            img = Image.fromarray(a)
        else:
            return
        doc.pil = img
        doc.push()
        # when effect applied, reset adjustments baseline to current image
        doc.reset_adjustments()
        self.refresh_controls_for_current()

    def rotate90(self):
        doc = self.current_doc()
        if not doc: return
        doc.pil = doc.pil.rotate(-90, expand=True)
        doc.push()
        self.refresh_controls_for_current()

    def flip_horizontal(self):
        doc = self.current_doc()
        if not doc: return
        doc.pil = ImageOps.mirror(doc.pil)
        doc.push()
        self.refresh_controls_for_current()

    def undo(self):
        doc = self.current_doc()
        if not doc: return
        ok = doc.undo()
        if not ok:
            QMessageBox.information(self, 'Bilgi', 'Geri alınacak işlem yok.')
        self.refresh_controls_for_current()

    def portrait_mode(self):
        doc = self.current_doc()
        if not doc: return
        img = doc.pil.copy()
        a = np.array(img)
        gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        faces = []
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        except Exception:
            faces = []
        arr = a.astype(np.float32)
        arr[:,:,0] *= 0.95
        arr[:,:,1] = np.minimum(255, arr[:,:,1]*1.05)
        arr[:,:,2] = np.minimum(255, arr[:,:,2]*1.05)
        a = arr.astype(np.uint8)
        if len(faces) > 0:
            for (x,y,w,h) in faces:
                roi = a[y:y+h, x:x+w]
                blurred = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
                a[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.6, blurred, 0.4, 0)
        else:
            a = cv2.bilateralFilter(a, d=9, sigmaColor=75, sigmaSpace=75)
        img2 = Image.fromarray(a)
        doc.pil = img2
        doc.push()
        self.refresh_controls_for_current()

    def auto_enhance(self):
        doc = self.current_doc()
        if not doc: return
        img = ImageEnhance.Contrast(doc.pil).enhance(1.08)
        img = ImageEnhance.Brightness(img).enhance(1.06)
        doc.pil = img
        doc.push()
        self.refresh_controls_for_current()

    def reset_adjustments(self):
        doc = self.current_doc()
        if not doc: return
        doc.reset_adjustments()
        # preview should use baseline history[-1]
        doc.pil = doc.history[-1].copy()
        self.refresh_controls_for_current()

    # ---------------------- Export ----------------------
    def export_current(self):
        doc = self.current_doc()
        if not doc:
            QMessageBox.information(self, 'Bilgi', 'Kayıt edilecek görsel yok.')
            return
        p = QFileDialog.getSaveFileName(self, 'Dışa Aktar', doc.path or os.getcwd(), 'PNG (*.png);;JPEG (*.jpg *.jpeg)')
        if p and p[0]:
            doc.pil.save(p[0])
            QMessageBox.information(self, 'Başarılı', 'Görsel kaydedildi.')

    def bulk_export(self):
        if not self.tabs_docs:
            QMessageBox.information(self, 'Bilgi', 'İşlenecek resim yok.')
            return
        folder = QFileDialog.getExistingDirectory(self, 'Klasör seç (Toplu Dışa Aktar)')
        if not folder:
            return
        for i, tab in enumerate(list(self.tabs_docs)):
            doc = tab.doc
            name = doc.path and os.path.basename(doc.path) or f'resim_{i+1}.png'
            dst = os.path.join(folder, f'processed_{i+1}_{name}')
            doc.pil.save(dst)
        QMessageBox.information(self, 'Bitti', f'{len(self.tabs_docs)} resim kaydedildi.')

# ---------------------- Run ----------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = PhotoEditorMain()
    win.show()
    sys.exit(app.exec_())
