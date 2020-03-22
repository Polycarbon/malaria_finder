import logging

from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QThread
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.image import resize_image, preprocess_image
import matplotlib.pyplot as plt
logger = logging.getLogger('data flow')


class CellDetector(QObject):
    onInitModelSuccess = QtCore.pyqtSignal()
    onDetectSuccess = QtCore.pyqtSignal(list, np.ndarray, list, list)

    def __init__(self, parent=None):
        super(CellDetector, self).__init__(parent)
        self.model = None
        self.thread = QThread()
        self.thread.start()
        self.moveToThread(self.thread)

    def initModel(self, path='src/resnet50.h5', backbone='resnet50'):
        logger.info('Initialize Model')
        if self.model is None:
            self.model = models.load_model(path, backbone_name=backbone)
        self.onInitModelSuccess.emit()
        logger.info('Init Model Success')

    def setObjectMap(self,objectmap):
        self.objectmap = objectmap

    def detect(self, key_list, ds, buffer):
        logger.info('Detecting Image')
        v_max = 11
        v_min = 2
        #preprocess
        frameDiff = np.abs(np.diff(buffer, axis=0))
        frameDiffSum = np.sum(frameDiff, axis=0)
        av = (frameDiffSum / len(frameDiff))
        av[av > v_max] = v_max
        av[av < v_min] = v_min
        # av = (frameDiffSum / len(frameDiff)).astype('uint8')
        normframe = (((av - v_min) / (v_max - v_min)) * 255).astype('uint8')
        # plt.imshow(normframe)
        # plt.show()
        image = np.stack((normframe,) * 3, axis=-1)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        cells = []
        sc = []
        for cell, score in zip(boxes[0], scores[0]):
            if score > 0.5:
                cells.append(tuple(cell))
                sc.append(score)
        self.onDetectSuccess.emit(key_list, ds, cells, sc)
        # for key in key_list:
        #     self.objectmap[key]['cells'] = cells
        #
        # logger.info('cell location : {}'.format(str(cells)))
