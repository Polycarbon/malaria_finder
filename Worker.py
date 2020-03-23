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
from imutils.video import FileVideoStream
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
        buffer.append(prev_gray.astype("int16"))
        # frame_count = 500
        if not os.path.exists(self.dataFileName):
            d = [(0, 0)]
            self.onFrameChanged.emit([0, 0])
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
                if dx < move_thres and dy < move_thres:
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
                        self.onBufferReady.emit(frameId - 1, buffer[-window_size:])
                    self.onBufferReady.emit(frameId, [])
                    # clear buffer
                    tmp = []
                    buffer = []
                # Extract rotation angle
                # da = np.arctan2(H[1, 0], H[0, 0])
                # Move to next frame
                prev_gray = curr_gray
                buffer.append(prev_gray.astype("int16"))
                self.onUpdateProgress.emit(frameId + 1, 'preprocess')
            if len(buffer) >= step_size:
                self.onBufferReady.emit(frameId, buffer[-window_size:])
            else:
                self.onBufferReady.emit(frameId, None)
            d = np.array(d)
            np.save(self.dataFileName, d)
        else:
            d = np.load(self.dataFileName)
            self.onFrameChanged.emit(d[0].tolist())
            tmp = []
            for frameId in range(1, frame_count):
                # Detect feature points in previous frame
                (dx, dy) = d[frameId]
                curr = self.fvs.read()
                self.onFrameChanged.emit([dx, dy])
                if dx < move_thres and dy < move_thres:
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
                        self.onBufferReady.emit(frameId - 1, buffer[-window_size:])
                    # clear buffer
                    tmp = []
                    buffer = []
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                prev_gray = curr_gray
                self.onUpdateProgress.emit(frameId + 1, 'preprocess')
            if len(buffer) >= step_size:
                self.onBufferReady.emit(frameId, buffer[-window_size:])
        logger.debug('preprocess finished')
        # self.onFrameChanged.emit(d)
        # kernelSize = int(fps * 2)
        # bx = np.abs(d[0]) > 0.5
        # by = np.abs(d[1]) > 0.5
        # bn = bx | by
        # bn = binary_closing(bn, structure=np.ones(kernelSize))
        # bn = bn[:frame_count]
        # # plt.figure(figsize=(20, 5))
        # # plt.plot(bn)
        # # plt.show()
        # inactive_intervals = find_inactive(bn)
        # for startId, endId in inactive_intervals:
        #     for start in range(startId, endId - step_size, step_size):
        #         key_list = list(range(start, start + window_size))
        #         if start + window_size > endId:
        #             d_buff = d[:, endId - window_size:endId]
        #             window = buffer[endId - window_size:endId]
        #         else:
        #             d_buff = d[:, start:start + window_size]
        #             window = buffer[start:start + window_size]
        #         # window = diff_frames[start:start + windowSize - 1]
        #         self.onBufferReady.emit(key_list, d_buff, window)
        # self.onFinish.emit()


class ObjectMappingThread(QThread):
    onUpdateProgress = QtCore.pyqtSignal(int, str)
    onUpdateObject = QtCore.pyqtSignal(defaultdict)

    def __init__(self, frame_count, fps):
        QThread.__init__(self)
        self.stopped_id = None
        self.frame_count = frame_count
        self.fps = fps
        self.window_size = fps*window_time
        self.objectmap = defaultdict(lambda: None)
        self.lastFrameId = 0
        self.currFrameId = 0
        self.Q = Queue()
        self.flow_list = []

    def __del__(self):
        self.wait()

    def updateOpticalFlow(self, d):
        self.flow_list.append(d)

    def queueOutput(self, *args):
        self.Q.put(args)
        logger.debug('{}-{} : queue success'.format(args[0], args[0] - 50))

    def run(self):
        while True:
            # otherwise, ensure the queue has room in it
            if not self.Q.empty():
                (lastest_id, cellObject, scores) = self.Q.get()

                # if
                # dx = ds[0] - ds[0, 0]
                # dy = ds[1] - ds[1, 0]
                # cellObject = np.array(cellObject)
                # if self.currentFrameId in frameIds:
                #     if len(cellObject) > 0:
                #         for fId, x, y in zip(frameIds, dx, dy):
                #             cellObject[:, [0, 2]] = cellObject[:, [0, 2]] + x
                #             cellObject[:, [1, 3]] = cellObject[:, [1, 3]] + y
                #             self.objectmap[fId] = {'area': None, 'cells': np.copy(cellObject), 'scores': scores}
                #     self.lastFrameId = self.currentFrameId
                #     self.lastCellObject = cellObject
                #     self.currentFrameId = frameIds[-1]
                #     self.onUpdateObject.emit(self.objectmap)
                # else:
                #     if len(self.lastCellObject) > 0:
                #         last_frameIds = list(range(self.lastFrameId, frameIds[0]))
                #         last_dx = self.ds[0, last_frameIds] - self.ds[0, last_frameIds][0]
                #         last_dy = self.ds[1, last_frameIds] - self.ds[1, last_frameIds][0]
                #         last_cells = self.objectmap[self.lastFrameId]['cells']
                #         for fId, x, y in zip(last_frameIds, last_dx, last_dy):
                #             last_cells[:, [0, 2]] = last_cells[:, [0, 2]] + x
                #             last_cells[:, [1, 3]] = last_cells[:, [1, 3]] + y
                #             self.objectmap[fId] = {'area': None, 'cells': np.copy(last_cells), 'scores': scores}
                #         self.lastFrameId = self.currentFrameId
                #         self.lastCellObject = last_cells
                #         self.currentFrameId = frameIds[-1]
                #         self.onUpdateObject.emit(self.objectmap)
                #
                logger.debug("progress {}/{} : cell{} - scores{}".format(lastest_id, self.stopped_id, str(cellObject),
                                                                         str(scores)))
                self.onUpdateProgress.emit(lastest_id + 1, 'objectMapping')
                if lastest_id == self.frame_count-1:
                    self.sleep(1)
                    return
