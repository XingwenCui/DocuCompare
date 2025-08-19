from PyQt5.QtWidgets import QLabel, QVBoxLayout, QDialog
from PyQt5.QtCore import Qt, pyqtSignal
from utils import cvimg_to_qpixmap

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cv_img = None
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

class ImagePreviewDialog(QDialog):
    def __init__(self, cv_img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图片预览")
        self.setGeometry(300, 100, 1000, 800)
        vbox, lbl = QVBoxLayout(), QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        pixmap = cvimg_to_qpixmap(cv_img)
        lbl.setPixmap(pixmap.scaled(1600, 1200, Qt.KeepAspectRatio))
        vbox.addWidget(lbl)
        self.setLayout(vbox)
