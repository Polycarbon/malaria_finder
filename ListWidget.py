import datetime

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QHBoxLayout, QLabel, QVBoxLayout, QLCDNumber
from PyQt5.uic import loadUi

from forms.Ui_WidgetItem import Ui_Form


class QCustomQWidget(QWidget):
    def __init__(self, parent=None):
        super(QCustomQWidget, self).__init__()
        self.ui = loadUi("forms/widgetitem.ui", self)
        self.setFixedSize(self.size())
        # super(QCustomQWidget, self).__init__(parent)
        # self.textQVBoxLayout = QVBoxLayout()
        # self.textUpQLabel = QLabel()
        # self.cellNumber = QLCDNumber()
        # self.cellNumber.setFixedWidth(70)
        # self.textQVBoxLayout.addWidget(self.textUpQLabel)
        # self.textQVBoxLayout.addWidget(self.cellNumber)
        # self.allQHBoxLayout = QHBoxLayout()
        # self.iconQLabel = QLabel()
        # self.allQHBoxLayout.addWidget(self.iconQLabel, 0)
        # self.allQHBoxLayout.addLayout(self.textQVBoxLayout, 1)
        # self.setLayout(self.allQHBoxLayout)
        # setStyleSheet
        # self.textUpQLabel.setStyleSheet('''
        #     color: rgb(0, 0, 255);
        # ''')
        # self.cellNumber.setStyleSheet('''
        #     color: rgb(255, 0, 0);
        # ''')

    def setCount(self, count):
        self.ui.cellNumber.display(count)

    def setTimeText(self, text):
        self.ui.timeLabel.setText(text)
        # self.textUpQLabel.setText(text)

    def setPreviewImg(self, qImg):
        self.ui.previewImgLabel.setPixmap(QPixmap(qImg))


class ListWidget(QWidget):

    def __init__(self, log, parent=None):
        super(ListWidget, self).__init__(parent)
        self.cellNumber = QtWidgets.QLCDNumber()
        # self.cellNumber.setGeometry(QtCore.QRect(90, 10, 71, 41))
        # self.cellNumber.setObjectName("cellNumber")
        self.timeLabel = QtWidgets.QLabel()
        # self.timeLabel.setGeometry(QtCore.QRect(10, 10, 71, 41))
        font = QFont()
        font.setPointSize(10)
        self.timeLabel.setFont(font)
        # self.timeLabel.setObjectName("timeLabel")
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.timeLabel, 0)
        self.layout.addWidget(self.cellNumber, 1)
        self.setLayout(self.layout)
