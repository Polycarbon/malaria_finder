# This Python file uses the following encoding: utf-8
import os

import cv2
import imutils
import numpy as np
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QImage, QKeySequence
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QStyle, QFileDialog, QListWidgetItem, QApplication, QStatusBar, \
    QShortcut
from PyQt5.uic import loadUi

import VideoInfo
from Detector import CellDetector
from ListWidget import QCustomQWidget
from ProcessDialog import ProcessDialog
from SaveDialog import SaveDialog
from VideoWidget import VideoWidget
from Worker import PreprocessThread, VideoWriterThread, ObjectMapper
import matplotlib.pyplot as plt

from mfutils import toQImage, drawBoxes, getHHMMSSFormat


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = loadUi('forms/mainwindow.ui', self)
        # self.ui = Ui_MainWindow()
        # self.ui.setupUi(self)
        self.input_name = None
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = VideoWidget(self)
        self.detector = CellDetector()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.videoWidget)
        self.ui.videoFrame.setLayout(layout)
        self.ui.playButton.setEnabled(False)
        self.ui.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.ui.playButton.clicked.connect(self.play)
        self.ui.openButton.clicked.connect(self.openFile)
        self.ui.saveButton.clicked.connect(self.saveFile)
        self.ui.saveButton.setEnabled(False)
        self.ui.timeSlider.sliderMoved.connect(self.setPosition)
        self.ui.modeCheckBox.stateChanged.connect(self.switchMode)
        self.ui.statusbar: QStatusBar
        self.ui.statusbar.showMessage("Init Model ...")
        # self.ui.statusbar.setLayout()
        self.mediaPlayer.setVideoOutput(self.videoWidget.videoSurface())
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        # self.mediaPlayer.videoAvailableChanged.connect(self.ui.listWidget.clear)
        self.mediaPlayer.setMuted(True)
        # shortcut
        QShortcut(QKeySequence('Space'), self).activated.connect(self.play)
        # s_max = self.maximumSize()
        # # self.ui.statusBar.setSizeGripEnabled(False)
        # self.show()
        # self.setFixedSize(s_max)

    def showEvent(self, *args, **kwargs):
        self.detector.onInitModelSuccess.connect(lambda: self.ui.statusbar.showMessage("Ready"))
        self.detector.initModel()

    def closeEvent(self, *args, **kwargs):
        QApplication.closeAllWindows()

    def switchMode(self, state):
        self.detector.setMode(state)
        if self.input_name:
            self.startProcess()

    def startProcess(self):
        if self.input_name:
            self.ui.listWidget.clear()
            self.sum_cells = 0
            self.log = []
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.input_name)))
            self.cap = cv2.VideoCapture(self.input_name)
            self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            VideoInfo.init(self.cap)
            dialog = ProcessDialog(self)
            dialog.setMaximum(self.frameCount)
            map_worker = ObjectMapper()
            map_worker.onUpdateObject.connect(self.updateObject)
            map_worker.onUpdateProgress.connect(dialog.updateProgress)
            ppc_worker = PreprocessThread(self.input_name)
            ppc_worker.onFrameChanged.connect(map_worker.updateOpticalFlow)
            ppc_worker.onFrameChanged.connect(self.detector.updateOpticalFlow)
            ppc_worker.onBufferReady.connect(self.detector.detect)
            ppc_worker.onUpdateProgress.connect(dialog.updateProgress)
            map_worker.onNewDetectedCells.connect(self.updateDetectLog)
            map_worker.finished.connect(dialog.close)
            dialog.closed.connect(ppc_worker.quit)
            dialog.closed.connect(map_worker.quit)
            self.detector.onDetectSuccess.connect(map_worker.queueOutput)
            # ppc_worker.finished.connect(dialog.close)
            # dialog.onReady2Read.connect(self.setOutput)
            dialog.show()
            map_worker.start()
            ppc_worker.start()
            dialog.exec_()
            self.ui.playButton.setEnabled(True)
            self.ui.saveButton.setEnabled(True)

    def openFile(self):
        file_name = QFileDialog.getOpenFileName(self, "Open Video")[0]
        if os.path.exists(file_name):
            self.input_name = file_name
            self.startProcess()

    def saveFile(self):
        dialog = SaveDialog()
        dialog.exec_()
        head, tail = os.path.split(self.input_name)
        file_prefix = tail.split('.')[0]
        worker = VideoWriterThread(self.cap, self.frame_objects, self.log, file_prefix,
                                   dialog.getSaveDirectory(),
                                   dialog.isSaveImage,
                                   dialog.isSaveVideo)
        worker.start()

    def updateObject(self, frame_objects):
        self.frame_objects = frame_objects
        duration = self.mediaPlayer.duration()
        self.videoWidget.setOutput(frame_objects, duration / self.frameCount)
        self.ui.playButton.setEnabled(True)
        self.ui.saveButton.setEnabled(True)

    def updateDetectLog(self, detected_frame_id, cell_map, cell_count):
        # append log
        widget = QCustomQWidget()
        self.sum_cells += cell_count
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, detected_frame_id)
        _, image = self.cap.read()
        _, min, sec = getHHMMSSFormat(self.mediaPlayer.duration() / self.frameCount * detected_frame_id)
        time_text = '{:02}-{:02}'.format(min, sec)
        self.log.append({"image": image.copy(), "detect_time": time_text, "cells": cell_map})
        drawBoxes(image, cell_map, (0, 255, 0))
        icon = imutils.resize(image, height=64)
        icon = toQImage(icon)
        widget.setPreviewImg(icon)
        widget.setCount(cell_count)
        widget.setDetectionFramePosition(self.mediaPlayer.duration() / self.frameCount * detected_frame_id)
        widget.setTimeText('{:02}:{:02}'.format(min, sec))
        widget.onDoubleClick.connect(self.setPosition)
        list_widget_item = QListWidgetItem(self.ui.listWidget)
        list_widget_item.setSizeHint(widget.size())
        self.ui.listWidget.addItem(list_widget_item)
        self.ui.listWidget.setItemWidget(list_widget_item, widget)
        self.ui.totalNumber.display(self.sum_cells)

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if state == QMediaPlayer.PlayingState:
            self.ui.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.ui.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.ui.timeSlider.setValue(position)
        duration = self.mediaPlayer.duration()
        _, dmin, dsec = getHHMMSSFormat(duration)
        _, pmin, psec = getHHMMSSFormat(position)
        self.ui.timeLabel.setText('{:02}:{:02}/{:02}:{:02}'.format(int(pmin), int(psec), int(dmin), int(dsec)))

    def durationChanged(self, duration):
        self.ui.timeSlider.setRange(0, duration)
        _, dmin, dsec = getHHMMSSFormat(duration)
        self.ui.timeLabel.setText('00:00/{:02}:{:02}'.format(int(dmin), int(dsec)))

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)
