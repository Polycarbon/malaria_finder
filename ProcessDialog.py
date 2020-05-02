# This Python file uses the following encoding: utf-8
import enum
from collections import defaultdict

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.uic import loadUi

from Worker import PreprocessThread
from forms import Ui_processdialog


class ProcessName(enum.Enum):
    Preprocess = 1
    ObjectMapping = 2


class ProcessDialog(QDialog):
    closed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(ProcessDialog, self).__init__(None, QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowTitleHint)
        self.ui = loadUi('forms/processdialog.ui', self)
        self.ui.buttonBox.rejected.connect(self.close)
        self.setWindowTitle('Finding')
        self.setWindowIcon(QIcon('src/ic_logo.png'))
        self.setFixedSize(self.size())
        self.progress1 = 0
        self.progress2 = 0

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()

    def readOutput(self, map, log):
        # print(out)
        self.onReady2Read.emit(map, log)
        self.close()

    def updateProgress(self, value, processname):
        if processname == 'preprocess':
            self.progress1 = value
        if processname == 'objectMapping':
            self.progress2 = value
        progress = (self.progress1+self.progress2)/2
        self.ui.progressBar.setValue(progress)

    def setMaximum(self, value):
        self.ui.progressBar.setMaximum(value)
        self.ui.progressBar.setMaximum(value)