import sys

import cv2
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QMainWindow, QListWidget, \
    QListWidgetItem
import numpy as np
import matplotlib.pyplot as plt

class QCustomQWidget (QWidget):
    def __init__ (self, parent = None):
        super(QCustomQWidget, self).__init__(parent)
        self.textQVBoxLayout = QVBoxLayout()
        self.textUpQLabel    = QLabel()
        self.textDownQLabel  = QLabel()
        self.textQVBoxLayout.addWidget(self.textUpQLabel)
        self.textQVBoxLayout.addWidget(self.textDownQLabel)
        self.allQHBoxLayout  = QHBoxLayout()
        self.iconQLabel      = QLabel()
        self.allQHBoxLayout.addWidget(self.iconQLabel, 0)
        self.allQHBoxLayout.addLayout(self.textQVBoxLayout, 1)
        self.setLayout(self.allQHBoxLayout)
        # setStyleSheet
        self.textUpQLabel.setStyleSheet('''
            color: rgb(0, 0, 255);
        ''')
        self.textDownQLabel.setStyleSheet('''
            color: rgb(255, 0, 0);
        ''')

    def setTextUp (self, text):
        self.textUpQLabel.setText(text)

    def setTextDown (self, text):
        self.textDownQLabel.setText(text)

    def setIcon (self, imagePath):
        self.iconQLabel.setPixmap(QPixmap(imagePath))

class exampleQMainWindow (QMainWindow):
    def __init__ (self):
        super(exampleQMainWindow, self).__init__()
        # Create QListWidget
        self.myQListWidget = QListWidget()
        for index, name, icon in [
            ('No.1', 'Meyoko',  'icon.png'),
            ('No.2', 'Nyaruko', 'icon.png'),
            ('No.3', 'Louise',  'icon.png')]:
            # Create QCustomQWidget
            myQCustomQWidget = QCustomQWidget()
            myQCustomQWidget.setTextUp(index)
            myQCustomQWidget.setTextDown(name)
            # Create QListWidgetItem
            myQListWidgetItem = QListWidgetItem(self.myQListWidget)
            # Set size hint
            myQListWidgetItem.setSizeHint(myQCustomQWidget.sizeHint())
            # Add QListWidgetItem into QListWidget
            self.myQListWidget.addItem(myQListWidgetItem)
            self.myQListWidget.setItemWidget(myQListWidgetItem, myQCustomQWidget)
        self.setCentralWidget(self.myQListWidget)

if __name__ == "__main__":
    def draw(rects, color):
        for l,t,w,h in rects:
            p1 = (l, t)
            p2 = (l +w, t + h)
            cv2.rectangle(ocv, p1, p2, color, 2)


    # draw image
    ocv = np.ones((400, 400, 3), np.uint8) * 127

    # demo array of 3 rects (the 1st 2 overlap and should be merged):
    rects = [[20, 20, 120, 130], [40, 40, 140, 100], [150, 150, 100, 100]]
    draw(rects, (0, 200, 0))

    # duplicate all of them ;)
    l = len(rects)
    for r in range(l):
        rects.append(rects[r])

    # min cluster size = 2, min distance = 0.5:
    rects, weights = cv2.groupRectangles(rects, 1, .5)
    draw(rects, (200, 0, 0))
    plt.imshow(ocv)
    plt.show()
    print(rects)
    print(weights)