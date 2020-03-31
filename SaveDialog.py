# This Python file uses the following encoding: utf-8
import enum
import os
from collections import defaultdict

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog
from PyQt5.uic import loadUi


class SaveDialog(QDialog):
    closed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(SaveDialog, self).__init__()
        self.ui = loadUi('forms/savedialog.ui', self)
        self.setWindowTitle('Save File')
        self.setFixedSize(self.size())
        self.ui.browseButton.clicked.connect(self.browseSaveDirectory)
        # self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()

    def browseSaveDirectory(self):
        out_dir = QFileDialog.getExistingDirectory(self, "Open Video")
        self.ui.pathlineEdit.setText(out_dir)

    def isSaveVideo(self):
        return self.ui.saveVideoCheckBox.isChecked()

    def isSaveImage(self):
        return self.ui.saveImageCheckBox.isChecked()

    def getSaveDirectory(self):
        return self.ui.pathlineEdit.text()