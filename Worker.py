import enum
import logging
import os
from queue import Queue

import cv2
from PyQt5.QtCore import QThread, QRect, QRectF, QPointF
from PyQt5 import QtCore
from PyQt5.QtGui import QPolygonF
from PyQt5.QtWidgets import QApplication
import numpy as np
from scipy.spatial.distance import cdist

from ObjectHandler import CellRect, ObjectTracker
from centroidtracker import CentroidTracker
from keras_retinanet import models
from scipy.ndimage import binary_closing
from scipy.spatial import distance
from skimage.filters import threshold_yen
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from imutils.video import FileVideoStream
from keras_retinanet.utils.image import preprocess_image, resize_image

logger = logging.getLogger('data flow')


def histEqualization_HSV(image):
    H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    return cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)


class VideoWriterThread(QThread):
    onUpdateProgress = QtCore.pyqtSignal(int)

    def __init__(self, cap, frame_objects, log, file_prefix, out_dir, save_image=True, save_video=True):
        QThread.__init__(self)
        self.cap = cap
        self.frame_objects = frame_objects
        self.log = log
        self.save_image = save_image
        self.save_video = save_video
        self.out_dir = out_dir
        self.file_prefix = file_prefix
        self.vid_file_name = os.path.join(self.out_dir, self.file_prefix + "_out.avi")
        logger.info("save video : {}".format(self.vid_file_name))

    def __del__(self):
        self.wait()

    def run(self):
        logger.info("start save video")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fwidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fheight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        flenght = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frate = int(self.cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(self.vid_file_name, fourcc, frate, (fwidth, fheight))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(0, flenght):
            ret, frame = self.cap.read()
            objects = self.frame_objects[i]["cells"]
            total_count = len(self.log[-1]['cells'].values())
            for id, obj in objects.items():
                cv2.rectangle(frame, (int(obj.left()), int(obj.top())),
                              (int(obj.right()), int(obj.bottom())),
                              (255, 0, 0), 2)
                cv2.putText(frame, 'id {}'.format(id), (int(obj.right()), int(obj.bottom())), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0))
            cv2.putText(frame, 'total count : ' + str(total_count), (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255))

            # self.onUpdateProgress.emit(i)
            out.write(frame)

        for log in self.log:
            image = log['image']
            detect_time = log['detect_time']
            cells = log['cells']
            for id, obj in cells.items():
                cv2.rectangle(image, (int(obj.left()), int(obj.top())),
                              (int(obj.right()), int(obj.bottom())),
                              (0, 255, 0), 2)
                cv2.putText(image, 'id {}'.format(id), (int(obj.right()), int(obj.bottom())), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0))
                cv2.putText(image, 'total count : ' + str(total_count), (10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255))
            file_name = os.path.join(self.out_dir, self.file_prefix + "_" + detect_time + ".png")
            cv2.imwrite(file_name, image)
        out.release()
        logger.info("save finish")


window_time = 2


class PreprocessThread(QThread):
    onImageReady = QtCore.pyqtSignal(list, list, list, np.ndarray)
    # onBufferReady = QtCore.pyqtSignal(list, np.ndarray, list)
    onBufferReady = QtCore.pyqtSignal(int, list)
    onFinish = QtCore.pyqtSignal()
    onUpdateProgress = QtCore.pyqtSignal(int, str)
    onFrameChanged = QtCore.pyqtSignal(list)

    def __init__(self, file_name):
        QThread.__init__(self)
        self.fvs = FileVideoStream(file_name).start()
        self.dataFileName = file_name[:-4] + '.npy'
        self.binarySignal = None

    def __del__(self):
        self.wait()

    def run(self):
        QApplication.processEvents()
        logger.info('start preprocess video')
        frame_count = int(self.fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = np.ceil(self.fvs.stream.get(cv2.CAP_PROP_FPS))
        window_size = int(fps * window_time)
        step_size = int(fps)
        move_thres = 0.5
        buffer = []
        prev = self.fvs.read()
        targetSize = (640, 360)
        # prev = cv2.resize(prev, targetSize)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        # prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        buffer.append(prev_gray.astype("int16"))
        # frame_count = 500
        if not os.path.exists(self.dataFileName):
            d = [(0, 0)]
            tmp = []
            for frameId in range(1, frame_count):
                # Detect feature points in previous frame
                prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                                   maxCorners=200,
                                                   qualityLevel=0.01,
                                                   minDistance=30,
                                                   blockSize=100)
                # Read next frame
                curr = self.fvs.read()
                # Convert to grayscale
                # curr = cv2.resize(curr, targetSize)
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                # curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
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
                d.append([dx, dy])
                self.onFrameChanged.emit([dx, dy])
                if abs(dx) < move_thres and abs(dy) < move_thres:
                    buffer.append(prev_gray.astype("int16"))
                    if len(buffer) == window_size:
                        # send buffer to predict
                        self.onBufferReady.emit(frameId, buffer[-window_size:])
                        # step buffer
                        tmp = buffer[:step_size]
                        buffer = buffer[step_size:]
                else:
                    tmp.extend(buffer)
                    if len(tmp) >= window_size:
                        self.onBufferReady.emit(frameId - 1, tmp[-window_size:])
                    # clear buffer
                    tmp = []
                    buffer = []
                # Extract rotation angle
                # da = np.arctan2(H[1, 0], H[0, 0])
                # Move to next frame
                prev_gray = curr_gray
                buffer.append(prev_gray.astype("int16"))
                self.onUpdateProgress.emit(frameId + 1, 'preprocess')
            self.onFrameChanged.emit([0, 0])
            if len(buffer) >= step_size:
                self.onBufferReady.emit(frameId, buffer[-window_size:])
            else:
                self.onBufferReady.emit(frameId, None)
            d = np.array(d)
            np.save(self.dataFileName, d)
        else:
            d = np.load(self.dataFileName)
            tmp = []
            for frameId in range(1, frame_count):
                # Detect feature points in previous frame
                (dx, dy) = d[frameId]
                curr = self.fvs.read()
                self.onFrameChanged.emit([dx, dy])
                if abs(dx) < move_thres and abs(dy) < move_thres:
                    buffer.append(prev_gray.astype("int16"))
                    if len(buffer) == window_size:
                        # send buffer to predict
                        self.onBufferReady.emit(frameId, buffer[-window_size:])
                        # step buffer
                        tmp = buffer[:step_size]
                        buffer = buffer[step_size:]
                else:
                    tmp.extend(buffer)
                    if len(tmp) >= window_size:
                        self.onBufferReady.emit(frameId - 1, tmp[-window_size:])
                    # clear buffer
                    tmp = []
                    buffer = []
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                prev_gray = curr_gray
                self.onUpdateProgress.emit(frameId + 1, 'preprocess')
            self.onFrameChanged.emit(d[0].tolist())
            if len(buffer) >= step_size:
                self.onBufferReady.emit(frameId, buffer[-window_size:])
        logger.debug('preprocess finished')


class ObjectMapper(QThread):
    onUpdateProgress = QtCore.pyqtSignal(int, str)
    onUpdateObject = QtCore.pyqtSignal(defaultdict)
    onNewDetectedCells = QtCore.pyqtSignal(int, OrderedDict, int)

    def __init__(self, frame_count, fps):
        QThread.__init__(self)
        self.stopped_id = None
        self.frame_count = frame_count
        self.fps = fps
        self.window_size = fps * window_time
        self.objectmap = defaultdict(lambda: {'area': None, 'cells': [], 'scores': []})
        self.last_cells = None
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.nextObjectID = 0
        self.countId = 0
        self.lastFrameId = 0
        self.currFrameId = 0
        self.Q = Queue()
        self.tracker = ObjectTracker()
        self.flow_list = []
        self.objectId = 0
        self.mask_area = None
        self.last_area = QRectF()

    def __del__(self):
        self.wait()

    def updateOpticalFlow(self, d):
        self.flow_list.append(d)

    def queueOutput(self, *args):
        self.Q.put(args)
        logger.debug('{}-{} : queue success'.format(args[0], args[0] - 50))

    def countInArea(self, area):
        new_count = 0
        for cell in self.objects.values():
            intersected = area.intersected(cell)
            intersected_ratio = (intersected.width() * intersected.height()) / (cell.height() * cell.width())
            if intersected_ratio > 0.8 and not cell.isCounted():
                cell.count(self.countId)
                self.countId += 1
                new_count += 1
        return new_count

    def run(self):
        while True:
            # otherwise, ensure the queue has room in it
            if not self.Q.empty():
                (end_id, c_area, detected_cells, scores) = self.Q.get()
                counting_area = QRectF(*c_area)
                counting_area = QPolygonF(counting_area)
                # if self.mask_area is None:
                #     self.mask_area = QPolygonF(QRectF(*c_area))
                # else:
                #     intersec_area = counting_area.intersected(self.mask_area)
                #     polygon_area()
                #     area_ratio = (intersec_area.width() * intersec_area.height()) / (
                #             counting_area.width() * counting_area.height())
                #     if area_ratio > 0.8:
                #         self.mask_area = QPolygonF(QRectF(*c_area))
                #     else:
                #         self.last_area = self.mask_area
                #         self.mask_area = QPolygonF(QRectF(*c_area))
                detected_cells = [CellRect(*cell, score=score) for cell, score in zip(detected_cells, scores)]
                start_id = int(end_id + 1 - self.window_size)
                if self.currFrameId < start_id:
                    # get last cells
                    # last_cells = self.objectmap[self.currFrameId]['cells']
                    for i in range(self.currFrameId, start_id):
                        x, y = self.flow_list[i]
                        # self.mask_area.translate(x, y)
                        cells = self.tracker.translated(x, y)
                        self.objectmap[i + 1] = {'area': None, 'cells': cells, 'scores': []}
                        self.onUpdateProgress.emit(i + 1, 'objectMapping')
                    self.currFrameId = end_id

                # last_cells = self.objectmap[self.currFrameId]["cells"]
                new_count = self.tracker.update(detected_cells)
                # new_count = self.countInArea(counting_area.united(self.last_area))
                if new_count > 0:
                    self.onNewDetectedCells.emit(self.currFrameId, self.objects, new_count)
                # new and last conflict
                cells = self.tracker.getObjects()
                self.objectmap[self.currFrameId] = {'area': counting_area, 'cells': cells,
                                                    'scores': scores}
                for i in range(start_id, end_id):
                    x, y = self.flow_list[i]
                    # self.mask_area.translate(x, y)
                    cells = self.tracker.translated(x, y)
                    self.objectmap[i + 1] = {'area': counting_area, 'cells': cells, 'scores': []}
                    self.onUpdateProgress.emit(i + 1, 'objectMapping')
                self.currFrameId = end_id
                self.onUpdateObject.emit(self.objectmap)
                logger.debug("progress {}/{} : cell{} - scores{}".format(end_id, self.frame_count, str(detected_cells),
                                                                         str(scores)))
                if end_id == self.frame_count - 1:
                    self.sleep(1)
                    return


def polygon_area(points):
    """Return the area of the polygon whose vertices are given by the
    sequence points.

    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return area / 2
