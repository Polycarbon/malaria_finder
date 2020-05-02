# This Python file uses the following encoding: utf-8
import enum
import os
from collections import defaultdict

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog, QVBoxLayout, QProgressBar, QDialogButtonBox
from PyQt5.uic import loadUi

import VideoInfo
from Worker import VideoWriterThread


class SaveDialog(QDialog):
    closed = QtCore.pyqtSignal()

    def __init__(self, cap, frame_objects, log, file_prefix, parent=None):
        super(SaveDialog, self).__init__()
        self.cap = cap
        self.frame_objects = frame_objects
        self.log = log
        self.file_prefix = file_prefix
        self.ui = loadUi('forms/savedialog.ui', self)
        self.setWindowTitle('Save File')
        self.setFixedSize(self.size())
        self.ui.browseButton.clicked.connect(self.browseSaveDirectory)
        self.ui.saveButton.clicked.connect(self.saveTask)
        # self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    def saveTask(self):
        self.progressBar = QProgressBar(self)
        self.progressBar.setMaximum(VideoInfo.FRAME_COUNT)
        self.ui.progressLayout.addWidget(self.progressBar)
        self.ui.saveButton.hide()
        worker = VideoWriterThread(self.cap, self.frame_objects, self.log, self.file_prefix,
                                   self.getSaveDirectory(),
                                   self.isSaveImage(),
                                   self.isSaveVideo())
        worker.onUpdateProgress.connect(self.updateProgress)
        worker.finished.connect(lambda :self.close())
        worker.start()

    def updateProgress(self, value):
        self.progressBar.setValue(value)

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