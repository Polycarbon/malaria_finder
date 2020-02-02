# This Python file uses the following encoding: utf-8
import sys
import os
from PyQt5.QtWidgets import QApplication
from MainWindow import MainWindow

os.chdir("C:/Users/Polycarbon/Desktop/Project/malaria detection device/malaria_finder")

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.setFixedSize(window.size())
    window.show()
    sys.exit(app.exec_())
