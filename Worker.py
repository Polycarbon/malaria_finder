import enum
import logging
import os
from queue import Queue

import cv2
from PyQt5.QtCore import QThread
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import numpy as np

from keras_retinanet import models
from scipy.ndimage import binary_closing
from scipy.spatial import distance
from skimage.filters import threshold_yen
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict

from keras_retinanet.utils.image import preprocess_image, resize_image

logger = logging.getLogger('data flow')


class ProcessName(enum.Enum):
    Preprocess = 1
    ObjectMapping = 2


class VideoWriterThread(QThread):
    onUpdateProgress = QtCore.pyqtSignal(int)

    def __init__(self, cap, map, file_name, mode='All'):
        QThread.__init__(self)
        self.cap = cap
        self.map = map
        self.file_name = file_name

    def __del__(self):
        self.wait()

    def run(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fwidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fheight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        flenght = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frate = int(self.cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(self.file_name, fourcc, frate, (fwidth, fheight))
        for i in range(0, flenght):
            ret, frame = self.cap.read()
            bnd = self.map[i]
            if bnd:
                top, left, bottom, right = bnd['area'].bbox
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(frame, 'count : ' + str(len(bnd['cell_bndbox'])), (10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255))
                for cell in bnd['cell_bndbox']:
                    top, left, bottom, right = cell.bbox
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            self.onUpdateProgress.emit(i)
            out.write(frame)
        out.release()


def find_inactive(bin_signal):
    diff_signal = np.diff(bin_signal.astype(int))
    gaps = []
    idx_down = 0
    for i, d in enumerate(diff_signal):
        if d > 0:
            gaps.append((idx_down, i))
            down = False
        if d < 0:
            idx_down = i + 1
            down = True
    if np.sum(bin_signal) != 0:
        if down:
            gaps.append((idx_down, i + 2))
    else:
        gaps.append((idx_down, len(bin_signal)))

    return np.array(gaps)


class PreprocessThread(QThread):
    onImageReady = QtCore.pyqtSignal(list, list, list, np.ndarray)
    onBufferReady = QtCore.pyqtSignal(list, np.ndarray, list)
    onFinish = QtCore.pyqtSignal()
    onUpdateProgress = QtCore.pyqtSignal(int, str)
    onFrameMove = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        QThread.__init__(self)
        self.cap = None
        self.dataFileName = None
        self.binarySignal = None

    def __del__(self):
        self.wait()

    def set_videoStream(self, file_name):
        self.cap = cv2.VideoCapture(file_name)
        self.dataFileName = file_name[:-4] + '.npy'

    def detect(self, image):
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        cells = []
        for cell, score in zip(boxes[0], scores[0]):
            if score > 0.1:
                cells.append(tuple(cell))

        return cells, scores[0]

    def movement_cell_locations(self, frame):
        try:
            thresh = threshold_yen(frame)
        except:
            thresh = threshold_yen(frame)
        binary = frame >= thresh
        closed = binary_closing(binary)
        # dilated = dilation(binary, square(5))
        label_img = label(closed)
        cellLocs = regionprops(label_img)
        cellLocs = [p for p in cellLocs if p.area > 100]
        return cellLocs

    def find_count_area(self, frame):
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = gray_img > 0.7 * 255
        closed = binary_closing(binary)
        eroded = dilation(closed, square(5))
        label_img = label(eroded, background=1)
        regions = regionprops(label_img)
        center = (int(gray_img.shape[0] / 2), int(gray_img.shape[1] / 2))
        area = min(regions, key=lambda props: distance.euclidean(center, props.centroid))
        # top, left, bottom, right = area.bbox
        # cv2.rectangle(gray_img, (left, top), (right, bottom), (0, 255, 0), 2)
        # plt.imshow(gray_img)
        # plt.show()
        return area

    #
    # def frame_crop(self,frame, area, window):
    #     top, left, bottom, right = area.bbox
    #     croped = (frame[top:bottom, left:right, 0]/50).astype('uint8')
    #     # croped = (frame[:, :, 0] / 50).astype('uint8')
    #     return croped
    #
    def cell_detect(self, image):
        cellLocs = self.movement_cell_locations(image)
        # plt.imshow(image)
        # plt.show()
        if len(cellLocs) > 10:
            cellLocs = []
        return cellLocs

    def run(self):
        QApplication.processEvents()
        logger.info('start preprocess video')
        frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = np.ceil(self.cap.get(cv2.CAP_PROP_FPS))
        diff_frames = np.empty((frameCount - 1, frameHeight, frameWidth), np.dtype('int16'))
        buffer = []
        # target_size = (64, 64)
        _, prev = self.cap.read()
        targetSize = (640, 360)
        # prev = cv2.resize(prev, targetSize)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        buffer.append(prev_gray.astype("int16"))
        frameCount = 500
        if not os.path.exists(self.dataFileName):
            dxs = [0]
            dys = [0]
            for frameId in range(1, frameCount):
                # Detect feature points in previous frame
                prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                                   maxCorners=200,
                                                   qualityLevel=0.01,
                                                   minDistance=30,
                                                   blockSize=100)
                # Read next frame
                success, curr = self.cap.read()
                if not success:
                    break
                # Convert to grayscale
                # curr = cv2.resize(curr, targetSize)
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                # Calculate optical flow (i.e. track feature points)
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
                # Sanity check
                assert prev_pts.shape == curr_pts.shape
                # Filter only valid points
                idx = np.where(status == 1)[0]
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]
                # Find transformation matrix
                H, inliers = cv2.estimateAffine2D(prev_pts, curr_pts)
                # Extract traslation
                dx = H[0, 2]
                dy = H[1, 2]
                dxs.append(dx)
                dys.append(dy)
                # Extract rotation angle
                da = np.arctan2(H[1, 0], H[0, 0])
                # Frame different
                # diff_frames[frameId - 1] = np.abs(curr_gray.astype("int16") - prev_gray.astype("int16"))
                # Move to next frame
                prev_gray = curr_gray
                buffer.append(prev_gray.astype("int16"))
                self.onUpdateProgress.emit(frameId, 'preprocess')

            d = np.array([dxs,dys])
            np.save(self.dataFileName, d)
        else:
            d = np.load(self.dataFileName)
            for frameId in range(0, frameCount):
                # Detect feature points in previous frame
                success, curr = self.cap.read()
                if not success:
                    break
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                buffer.append(curr_gray.astype("int16"))
                self.onUpdateProgress.emit(frameId, 'preprocess')

        self.onFrameMove.emit(d)
        kernelSize = int(fps * 2)
        bx = np.abs(d[0]) > 0.5
        by = np.abs(d[1]) > 0.5
        bn = bx | by
        bn = binary_closing(bn, structure=np.ones(kernelSize))
        bn = bn[:frameCount]
        # plt.figure(figsize=(20, 5))
        # plt.plot(bn)
        # plt.show()
        inactive_intervals = find_inactive(bn)
        windowSize = int(fps * 2)
        stepSize = int(fps)
        for startId, endId in inactive_intervals:
            for start in range(startId, endId - stepSize, stepSize):
                key_list = list(range(start, start + windowSize))
                if start + windowSize > endId:
                    d_buff = d[:, endId - windowSize:endId]
                    window = buffer[endId - windowSize:endId]
                else:
                    d_buff = d[:, start:start + windowSize]
                    window = buffer[start:start + windowSize]
                # window = diff_frames[start:start + windowSize - 1]
                self.onBufferReady.emit(key_list, d_buff, window)
                # sum = np.sum(window, axis=0)
                # av = (sum / windowSize)
                # av[av > v_max] = v_max
                # av[av < v_min] = v_min
                # av = (sum / len(window)).astype('uint8')
                # normframe = (((av - v_min) / (v_max - v_min)) * 255).astype('uint8')
                # image = np.stack((normframe,) * 3, axis=-1)
                # key_list = list(range(start, start + windowSize))
                # self.onImageReady.emit(key_list, image)
                # logger.debug('{} - {} lenght = {}'.format(start, start + windowSize - 1, window.shape))

        # kernel = (64, 64)
        # diff_frame_ratio = np.sum(diff_frames[:, :kernel[0], :kernel[1]], axis=(1, 2)) / (kernel[0] * kernel[1])
        # bin_signal = diff_frame_ratio > 2.75
        # bin_signal = binary_closing(bin_signal, structure=np.ones(40))
        # gaps = find_inactive(bin_signal)
        # log = []
        # v_max = 11
        # v_min = 2
        # if len(gaps) != 0:
        #     for gap in gaps:
        #         lenght = gap[1] - gap[0]
        #         n_vote = 5
        #         step_size = int(lenght / (n_vote - 1))
        #         window_size = 50
        #         sum_size = window_size - 1
        #         self.cap.set(cv2.CAP_PROP_POS_FRAMES, gap[0])
        #         _, frame = self.cap.read()
        #         area = self.find_count_area(frame)
        #         key_list = list(range(gap[0],gap[1]))
        #         map = zip(key_list, [{'area': area}]*lenght)
        #         self.objectmap.update(dict(map))
        #         window = diff_frames[gap[0]:gap[0] + window_size]
        #         sumframe = np.sum(window, axis=0)
        #         aveframe = (sumframe / sum_size + 1)
        #         aveframe[aveframe > v_max] = v_max
        #         aveframe[aveframe < v_min] = v_min
        #         aveframe = (sumframe / len(window)).astype('uint8')
        #         normframe = (((aveframe - v_min) / (v_max - v_min)) * 255).astype('uint8')
        #         image = np.stack((normframe,) * 3, axis=-1)
        #         self.onImageReady.emit(key_list, image)
        # #         print(cell_locs)
        # #         for i in range(gap[0], gap[1] + 1):
        # #             o = {'area': area, 'cell_bndbox': cell_locs}
        # #             map[i] = o
        # #         log.append({'min_time': gap[0], 'max_time': gap[1], 'cells_count': len(cell_locs)})
        # #         # for cell in cellLocs:
        # #         #     top, left, bottom, right = cell.bbox
        # #         #     output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # #         #     output = cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)
        # #         #     plt.imshow(output)
        # #         #     plt.show()
        # #
        self.onFinish.emit()


class ObjectMappingThread(QThread):
    onUpdateProgress = QtCore.pyqtSignal(int, str)
    onUpdateObject = QtCore.pyqtSignal(defaultdict)

    def __init__(self):
        QThread.__init__(self)
        self.objectmap = defaultdict(lambda: None)
        self.stopped = False
        self.lastFrameId = 0
        self.currentFrameId = 0
        self.Q = Queue()

    def __del__(self):
        self.wait()

    def setFrameMove(self, ds):
        self.ds = ds

    def set_videoStream(self, file_name):
        self.cap = cv2.VideoCapture(file_name)

    def queueCellObjects(self, *args):
        self.Q.put(args)
        logger.debug('{}-{} : queue success'.format(args[0][1], args[0][-1]))

    def run(self):
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.empty():
                (frameIds, ds, cellObject, scores) = self.Q.get()
                dx = ds[0] - ds[0, 0]
                dy = ds[1] - ds[1, 0]
                cellObject = np.array(cellObject)
                if self.currentFrameId in frameIds:
                    if len(cellObject) > 0:
                        for fId, x, y in zip(frameIds, dx, dy):
                            cellObject[:, [0, 2]] = cellObject[:, [0, 2]] + x
                            cellObject[:, [1, 3]] = cellObject[:, [1, 3]] + y
                            self.objectmap[fId] = {'area': None, 'cells': np.copy(cellObject), 'scores': scores}
                    self.lastFrameId = self.currentFrameId
                    self.lastCellObject = cellObject
                    self.currentFrameId = frameIds[-1]
                    self.onUpdateObject.emit(self.objectmap)
                else:
                    if len(self.lastCellObject) > 0:
                        last_frameIds = list(range(self.lastFrameId,frameIds[0]))
                        last_dx = self.ds[0, last_frameIds]-self.ds[0, last_frameIds][0]
                        last_dy = self.ds[1, last_frameIds]-self.ds[1, last_frameIds][0]
                        last_cells = self.objectmap[self.lastFrameId]['cells']
                        for fId, x, y in zip(last_frameIds, last_dx, last_dy):
                            last_cells[:, [0, 2]] = last_cells[:, [0, 2]] + x
                            last_cells[:, [1, 3]] = last_cells[:, [1, 3]] + y
                            self.objectmap[fId] = {'area': None, 'cells': np.copy(last_cells), 'scores': scores}
                        self.lastFrameId = self.currentFrameId
                        self.lastCellObject = last_cells
                        self.currentFrameId = frameIds[-1]
                        self.onUpdateObject.emit(self.objectmap)

                logger.debug("lid {} - cell{} - scores{}".format(frameIds[-1], str(cellObject), str(scores)))
                self.onUpdateProgress.emit(frameIds[-1], 'objectMapping')
