import logging

import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QThread
import numpy as np
from scipy.spatial import distance
from skimage import feature, exposure
from skimage.feature import blob_dog, blob_doh
from skimage.filters import threshold_yen
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, dilation, square, erosion

import VideoInfo
from LineHandler import extractLines, calculateBoundingPoints, extend_verticals, extend_horizontals
from keras_retinanet import models
from keras_retinanet.utils.image import resize_image, preprocess_image
import matplotlib.pyplot as plt

logger = logging.getLogger('data flow')

IMAGE_PROCESSING = "Image Processing"
DEEP_LEARNING = "Deep Learning"
MODES = [DEEP_LEARNING, IMAGE_PROCESSING]
BLOB = 2


class CellDetector(QObject):
    onInitModelSuccess = QtCore.pyqtSignal()
    onDetectSuccess = QtCore.pyqtSignal(int, list, list, list)

    def __init__(self, parent=None):
        super(CellDetector, self).__init__(parent)
        self.model = None
        self.thread = QThread()
        self.thread.start()
        self.moveToThread(self.thread)
        self.mode = DEEP_LEARNING

    def initModel(self, path='src/resnet50.h5', backbone='resnet50'):
        if self.mode == DEEP_LEARNING:
            logger.info('Initialize Model')
            if self.model is None:
                self.model = models.load_model(path, backbone_name=backbone)
                self.onInitModelSuccess.emit()
                logger.info('Init Model Success')
        else:
            logger.info('no need to initialize Model')
            self.onInitModelSuccess.emit()

    def setMode(self, mode):
        if mode == 1:
            self.mode = DEEP_LEARNING
        else:
            self.mode = IMAGE_PROCESSING

    def setObjectMap(self, objectmap):
        self.objectmap = objectmap


    @staticmethod
    def find_count_area(gray):
        binary = gray > 0.7 * 255
        closed = binary_closing(binary, square(5))

        dialated = dilation(closed, square(30))
        eroded = erosion(dialated, square(20))
        label_img = label(eroded, background=1)
        regions = regionprops(label_img)
        center = (int(gray.shape[0] / 2), int(gray.shape[1] / 2))
        area = min(regions, key=lambda props: distance.euclidean(center, props.centroid))

        t, l, b, r = area.bbox
        margin = 10
        l = l - margin
        t = t - margin
        r = r + margin
        b = b + margin
        # return polygon point vector
        return [(l, t), (r, t), (r, b), (l, b), (l, t)]

    def detect(self, cur_frame_id, buffer):
        v_max = 10
        v_min = 1
        # flow_list = np.array(self.flow_list[cur_frame_id - 50:cur_frame_id]).transpose()
        # move_distances = np.sum(np.sqrt(flow_list[0] ** 2 + flow_list[1] ** 2))
        # if move_distances > 5:
        #     self.onDetectSuccess.emit(cur_frame_id,[], [], [])
        #     return
        verticals, horizontals = extractLines(buffer[0], threshold=0.66)
        shape = buffer[0].shape
        center = (int(shape[1] / 2), int(shape[0] / 2))
        x_bound = [0, shape[1]]
        y_bound = [0, shape[0]]
        vs = extend_verticals(verticals, x_bound, y_bound)
        hs = extend_horizontals(horizontals, x_bound, y_bound)
        area_vec = calculateBoundingPoints(center, vs, hs)
        # area_vec = self.find_count_area(buffer[0])

        frameDiff = np.abs(np.diff(buffer, axis=0))
        frameDiffSum = np.sum(frameDiff, axis=0)
        av = (frameDiffSum / len(frameDiff))
        av[av > v_max] = v_max
        av[av < v_min] = v_min
        normframe = (((av - v_min) / (v_max - v_min)) * 255).astype('uint8')
        if self.mode == DEEP_LEARNING:
            # preprocess
            image = np.stack((normframe,) * 3, axis=-1)
            image = preprocess_image(image)
            image, scale = resize_image(image)
            boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes /= scale
            cells = []
            sc = []
            for cell, score in zip(boxes[0], scores[0]):
                if score > 0.5:
                    l, t, r, b = cell
                    cells.append([int(l), int(t), int(r - l), int(b - t)])
                    # cells.append(cell)
                    sc.append(score)
            # min cluster size = 2, min distance = 0.5:
            cells.extend(cells)
            cells, weights = cv2.groupRectangles(cells, 1, 1.0)
            if len(cells) == 0:
                self.onDetectSuccess.emit(cur_frame_id, area_vec, [], [])
                return
            self.onDetectSuccess.emit(cur_frame_id, area_vec, cells.tolist(), sc)

        if self.mode == IMAGE_PROCESSING:
            image = normframe
            thresh = threshold_yen(image)
            binary = image >= thresh
            closed = binary_closing(binary)
            # plt.title("move distance:{}".format(move_distances))
            # plt.imshow(closed)
            # plt.show()
            label_img = label(closed)
            cell_locs = regionprops(label_img)
            # tlbr to ltrb
            cells = []
            for cell in cell_locs:
                t, l, b, r = cell.bbox
                prop = (b-t)/(r-l)
                if 100 < cell.area and 0.2 < prop < 6.0:
                    cells.append([l, t, r - l, b - t])
            if len(cells) > 5:
                # print("num{} move{}".format(len(cells),move_distances))
                self.onDetectSuccess.emit(cur_frame_id, area_vec, [], [])
                return
            cells.extend(cells)
            cells, weights = cv2.groupRectangles(cells, 1, 1.0)
            sc = [1.0] * len(cells)
            self.onDetectSuccess.emit(cur_frame_id, area_vec, list(cells), sc)

        if self.mode == BLOB:
            image = normframe
            blobs = blob_doh(image, min_sigma=20, threshold=.1)
            cells = []
            for blob in blobs:
                y, x, r = blob
                if r > 10:
                    cells.append([x - r, y - r, r, r])
            cells.extend(cells)
            cells, weights = cv2.groupRectangles(cells, 1, 1.0)
            if len(cells) > 5:
                # print("num{} move{}".format(len(cells),move_distances))
                self.onDetectSuccess.emit(cur_frame_id, area_vec, [], [])
                return
            sc = [1.0] * len(cells)
            self.onDetectSuccess.emit(cur_frame_id, area_vec, list(cells), sc)
