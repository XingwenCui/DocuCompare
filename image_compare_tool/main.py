import sys
from PyQt5.QtWidgets import QApplication
from gui import ImageCompareApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageCompareApp()
    window.show()
    sys.exit(app.exec_())
