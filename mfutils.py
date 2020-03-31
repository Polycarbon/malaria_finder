import cv2
import numpy as np
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QImage


def toQImage(image: np.ndarray):
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    return QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()


def drawBoxes(image: np.ndarray, boxes: dict, color, thickness=2):
    for id, box in boxes.items():
        cv2.rectangle(image,
                      (int(box.left()), int(box.top())),
                      (int(box.right()), int(box.bottom())), color)


def getHHMMSSFormat(ms):
    hour, remainder = divmod(ms, 3600000)
    min, remainder = divmod(remainder, 60000)
    sec, _ = divmod(remainder, 1000)
    return int(hour), int(min), int(sec)
