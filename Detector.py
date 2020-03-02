from PyQt5 import QtCore
from PyQt5.QtCore import QObject
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import resize_image, preprocess_image


class CellDetector(QObject):
    onInitModelSuccess = QtCore.pyqtSignal()
    onDetectSuccess = QtCore.pyqtSignal(list, list)

    def __init__(self,parent = None):
        super(CellDetector, self).__init__(parent)
        self.model = None

    def initModel(self,path='src/resnet50.h5',backbone='resnet50'):
        if self.model is None:
            self.model = models.load_model(path, backbone_name=backbone)
        self.onInitModelSuccess.emit()

    def detect(self,image):
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        cells = []
        sc = []
        for cell, score in zip(boxes[0], scores[0]):
            if score > 0.1:
                cells.append(tuple(cell))
                sc.append(score)
        self.onDetectSuccess.emit(cells, sc)

