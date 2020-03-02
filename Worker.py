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
            if bnd :
                top, left, bottom, right = bnd['area'].bbox
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(frame, 'count : '+str(len(bnd['cell_bndbox'])), (10, 30),
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
            gaps.append([idx_down, i])
            down = False
        if d < 0:
            idx_down = i + 1
            down = True
    if np.sum(bin_signal) != 0:
        if down:
            gaps.append([idx_down, i + 2])
    else:
        gaps.append([idx_down, len(bin_signal)])

    return np.array(gaps)


class DetectionThread(QThread):
    onFinish = QtCore.pyqtSignal(defaultdict, list)
    onUpdateProgress = QtCore.pyqtSignal(int)

    def __init__(self, cap):
        QThread.__init__(self)
        self.cap = cap

    def __del__(self):
        self.wait()

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
        self.model = models.load_model('src/resnet50.h5', backbone_name='resnet50')
        QApplication.processEvents()
        frame_lenght = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_size = (frameHeight, frameWidth)
        # target_size = (64, 64)
        ret, frame = self.cap.read()
        frame_id = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # resize = cv2.resize(gray, target_size).astype("int16")
        resize = gray
        prev_frame = resize
        window_size = 4000
        diff_frames = np.empty((frame_lenght - 1, *target_size), np.dtype('int16'))
        while frame_id < frame_lenght - 1:
            ret, frame = self.cap.read()

            frame_id += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # resize = cv2.resize(gray, target_size).astype("int16")
            resize = gray.astype("int16")
            diff_frames[frame_id - 1] = np.abs(resize - prev_frame)

            prev_frame = resize

            self.onUpdateProgress.emit(frame_id)
        kernel = (64, 64)
        diff_frame_ratio = np.sum(diff_frames[:, :kernel[0], :kernel[1]], axis=(1, 2)) / (kernel[0] * kernel[1])
        bin_signal = diff_frame_ratio > 2.75
        bin_signal = binary_closing(bin_signal, structure=np.ones(40))
        gaps = find_inactive(bin_signal)
        map = defaultdict(lambda: None)
        log = []
        v_max = 11
        v_min = 2
        if len(gaps) != 0:
            for gap in gaps:
                lenght = gap[1]-gap[0]
                n_vote = 5
                step_size = int(lenght/(n_vote-1))
                window_size = 50
                sum_size = window_size-1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, gap[0])
                _, frame = self.cap.read()
                area = self.find_count_area(frame)
                window = diff_frames[gap[0]:gap[0]+window_size]
                sumframe = np.sum(window, axis=0)
                aveframe = (sumframe/sum_size+1)
                aveframe[aveframe > v_max] = v_max
                aveframe[aveframe < v_min] = v_min
                aveframe = (sumframe / len(window)).astype('uint8')
                normframe = (((aveframe - v_min) / (v_max - v_min)) * 255).astype('uint8')
                image = np.stack((normframe,) * 3, axis=-1)
                cell_locs, scores = self.detect(image)
                print(cell_locs)
                for i in range(gap[0], gap[1] + 1):
                    o = {'area': area, 'cell_bndbox': cell_locs}
                    map[i] = o
                log.append({'min_time': gap[0], 'max_time': gap[1], 'cells_count': len(cell_locs)})
                # for cell in cellLocs:
                #     top, left, bottom, right = cell.bbox
                #     output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #     output = cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)
                #     plt.imshow(output)
                #     plt.show()

        self.onFinish.emit(map, log)
