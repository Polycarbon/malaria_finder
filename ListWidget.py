import datetime

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QHBoxLayout, QLabel, QVBoxLayout, QLCDNumber
from PyQt5.uic import loadUi

from forms.Ui_WidgetItem import Ui_Form


class QCustomQWidget(QWidget):
    def __init__(self, parent=None):
        super(QCustomQWidget, self).__init__(parent)
        self.textQVBoxLayout = QVBoxLayout()
        self.textUpQLabel = QLabel()
        # self.textDownQLabel = QLabel()
        self.textQVBoxLayout.addWidget(self.textUpQLabel)
        # self.textQVBoxLayout.addWidget(self.textDownQLabel)
        self.allQHBoxLayout = QHBoxLayout()
        # self.iconQLabel = QLabel()
        self.cellNumber = QLCDNumber()
        self.cellNumber.setFixedWidth(70)
        # self.allQHBoxLayout.addWidget(self.iconQLabel, 0)
        self.allQHBoxLayout.addLayout(self.textQVBoxLayout, 1)
        self.allQHBoxLayout.addWidget(self.cellNumber, 1)
        self.setLayout(self.allQHBoxLayout)
        # setStyleSheet
        self.textUpQLabel.setStyleSheet('''
            color: rgb(0, 0, 255);
        ''')
        # self.textDownQLabel.setStyleSheet('''
        #     color: rgb(255, 0, 0);
        # ''')

    def setCount(self, count):
        self.cellNumber.display(count)

    def setTimeText(self, fram_id):
        ms = fram_id * 33

        d = datetime.timedelta(milliseconds=int(ms))
        if d.seconds > 0:
            self.textUpQLabel.setText(str(d).split('.')[0])
        else:
            self.textUpQLabel.setText('0:00:00')
        # self.textUpQLabel.setText(text)

    def setTextDown(self, text):
        self.textDownQLabel.setText(text)

    def setIcon(self, imagePath):
        self.iconQLabel.setPixmap(QPixmap(imagePath))


class ListWidget(QWidget):

    def __init__(self, log, parent=None):
        super(ListWidget, self).__init__(parent)
        self.cellNumber = QtWidgets.QLCDNumber()
        # self.cellNumber.setGeometry(QtCore.QRect(90, 10, 71, 41))
        # self.cellNumber.setObjectName("cellNumber")
        self.timeLabel = QtWidgets.QLabel()
        # self.timeLabel.setGeometry(QtCore.QRect(10, 10, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.timeLabel.setFont(font)
        # self.timeLabel.setObjectName("timeLabel")
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.timeLabel, 0)
        self.layout.addWidget(self.cellNumber, 1)
        self.setLayout(self.layout)
