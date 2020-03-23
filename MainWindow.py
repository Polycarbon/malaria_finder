# This Python file uses the following encoding: utf-8
import os
from collections import defaultdict

import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QStyle, QFileDialog, QListWidgetItem, QApplication
from PyQt5.uic import loadUi
from imutils.video import count_frames
from DetectedObject import ObjectMap
from Detector import CellDetector
from ListWidget import ListWidget, QCustomQWidget
from ProcessDialog import ProcessDialog
from VideoWidget import VideoWidget
from Worker import PreprocessThread, VideoWriterThread, ObjectMappingThread
from forms.Ui_mainwindow import Ui_MainWindow
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = loadUi('forms/mainwindow.ui', self)
        # self.ui = Ui_MainWindow()
        # self.ui.setupUi(self)
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
        self.ui.statusbar.showMessage("Init Model ...")
        self.mediaPlayer.setVideoOutput(self.videoWidget.videoSurface())
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        # s_max = self.maximumSize()
        # # self.ui.statusBar.setSizeGripEnabled(False)
        # self.show()
        # self.setFixedSize(s_max)

    def showEvent(self, *args, **kwargs):
        self.detector.onInitModelSuccess.connect(lambda: self.ui.statusbar.showMessage("Ready"))
        self.detector.initModel()

    def closeEvent(self, *args, **kwargs):
        QApplication.closeAllWindows()

    def openFile(self):
        file_name = QFileDialog.getOpenFileName(self, "Open Video")[0]
        if os.path.exists(file_name):
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
            self.cap = cv2.VideoCapture(file_name)
            self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = np.ceil(self.cap.get(cv2.CAP_PROP_FPS))
            window_time = 2  # sec
            dialog = ProcessDialog(self)
            dialog.setMaximum(self.frameCount)
            map_worker = ObjectMappingThread(self.frameCount, fps)
            map_worker.onUpdateObject.connect(self.updateObject)
            map_worker.onUpdateProgress.connect(dialog.updateProgress)
            ppc_worker = PreprocessThread(file_name)
            ppc_worker.onFrameChanged.connect(map_worker.updateOpticalFlow)
            ppc_worker.onBufferReady.connect(self.detector.detect)
            ppc_worker.onUpdateProgress.connect(dialog.updateProgress)
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

    def saveFile(self):
        worker = VideoWriterThread(self.cap, self.map, 'output.avi')
        dialog = ProcessDialog()
        dialog.setMaximum(self.frameCount)
        worker.onUpdateProgress.connect(dialog.updateProgress)
        worker.finished.connect(dialog.close)
        dialog.show()
        worker.start()
        dialog.exec_()

    def updateObject(self, frame_objects):
        duration = self.mediaPlayer.duration()
        self.videoWidget.setOutput(frame_objects, duration / self.frameCount)
        self.ui.playButton.setEnabled(True)
        self.ui.saveButton.setEnabled(True)

    def setOutput(self):
        duration = self.mediaPlayer.duration()
        self.videoWidget.setOutput(self.objectmap, duration / self.frameCount)

        # sum = 0
        # for log in logs:
        #     # Create QCustomQWidget
        #     myQCustomQWidget = QCustomQWidget()
        #     sum += log['cells_count']
        #     myQCustomQWidget.setCount(log['cells_count'])
        #     myQCustomQWidget.setTimeText(log['min_time'])
        #     # Create QListWidgetItem
        #     myQListWidgetItem = QListWidgetItem(self.ui.listWidget)
        #     # Set size hint
        #     myQListWidgetItem.setSizeHint(myQCustomQWidget.sizeHint())
        #     # Add QListWidgetItem into QListWidget
        #     self.ui.listWidget.addItem(myQListWidgetItem)
        #     self.ui.listWidget.setItemWidget(myQListWidgetItem, myQCustomQWidget)
        # self.ui.totalNumber.display(sum)

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

    def durationChanged(self, duration):
        self.ui.timeSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)
