# This Python file uses the following encoding: utf-8
import enum
from collections import defaultdict

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.uic import loadUi

from Worker import PreprocessThread
from forms import Ui_processdialog


class ProcessName(enum.Enum):
    Preprocess = 1
    ObjectMapping = 2


class ProcessDialog(QDialog):

    def __init__(self):
        super(ProcessDialog, self).__init__()
        self.ui = loadUi('forms/processdialog.ui', self)
        self.ui.buttonBox.rejected.connect(self.close)
        self.setWindowTitle('Processing')
        self.setFixedSize(self.size())
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    # def closeEvent(self, event):
    #     if self.worker.isRunning():
    #         reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
    #                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    #         if reply == QMessageBox.Yes:
    #             event.accept()
    #         else:
    #             event.ignore()

    def readOutput(self, map, log):
        # print(out)
        self.onReady2Read.emit(map, log)
        self.close()

    def updateProgress(self, value, processname):
        if processname == 'preprocess':
            self.ui.progressBar1.setValue(value)
        if processname == 'objectMapping':
            self.ui.progressBar2.setValue(value)

    def setMaximum(self, value):
        self.ui.progressBar1.setMaximum(value)
        self.ui.progressBar2.setMaximum(value)