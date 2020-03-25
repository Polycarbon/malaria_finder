# This Python file uses the following encoding: utf-8
import logging
import sys
import os
from PyQt5.QtWidgets import QApplication
from MainWindow import MainWindow

os.chdir("C:/Users/Polycarbon/Desktop/Project/malaria detection device/malaria_finder")
styleData="""
QWidget
{
    color: #b1b1b1;
    background-color: #323232;
}
QSlider::groove:horizontal {
    background-color:#242424;
    border: 1px solid #242424;
    height: 10px;
    margin: 0px;
}
QSlider::handle:horizontal {
    background-color: #ff5860;
    border: 2px solid #FFFFFF;
    border-radius: 5px; 
    width: 20px;
    margin: -5px 0px;
}
QSlider::add-page:qlineargradient {
    background: grey;
    border-top-right-radius: 9px;
    border-bottom-right-radius: 9px;
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
}

QSlider::sub-page:qlineargradient {
    background: red;
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
    border-top-left-radius: 9px;
    border-bottom-left-radius: 9px;
}
QProgressBar
{
    border: 2px solid grey;
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk
{
    background-color: #d7801a;
    width: 2.15px;
    margin: 0.5px;
}
QPushButton:pressed
{
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);
}
QComboBox:hover,QPushButton:hover
{
    border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
}
QPushButton
{
    color: #b1b1b1;
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);
    border-width: 1px;
    border-color: #1e1e1e;
    border-style: solid;
    border-radius: 6;
    padding: 3px;
    font-size: 12px;
    padding-left: 5px;
    padding-right: 5px;
}"""

logging.basicConfig(format="%(threadName)s:%(message)s")
logger = logging.getLogger('data flow')
logger.setLevel(logging.DEBUG)
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.setStyleSheet(styleData)
    window.setFixedSize(window.size())
    window.show()
    sys.exit(app.exec_())
