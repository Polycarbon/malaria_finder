# %%
from collections import defaultdict
from scipy.ndimage import binary_closing
from scipy.spatial import distance
from skimage.filters import threshold_yen
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square
from scipy import stats
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# %%
file_name = 'src/manual_1.MOV'
# file_name = 'src/1-1.mp4'
# file_name = 'src/m1.avi'
cap = cv2.VideoCapture(file_name)
frame_lenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# target_size = (frameHeight, frameWidth)
# %%%
frame_id = 0
prev_frame = None
diff_frames = np.empty((frame_lenght - 1, *(frameHeight, frameWidth)), np.dtype('int16'))
# %%%
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resize = gray.astype("int16")
        prev_frame = resize
        frame_id += 1
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resize = gray.astype("int16")
        diff_frames[frame_id - 1] = np.abs(resize - prev_frame)
        prev_frame = resize
        frame_id += 1


# %%
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


def find_count_area(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = gray_img > 0.7 * 255
    closed = binary_closing(binary)
    eroded = dilation(closed, square(5))
    label_img = label(eroded, background=1)
    regions = regionprops(label_img)
    center = (int(gray_img.shape[0] / 2), int(gray_img.shape[1] / 2))
    area = min(regions, key=lambda props: distance.euclidean(center, props.centroid))
    return area


kernel = (64, 64)
diff_frame_ratio = np.sum(diff_frames[:, :kernel[0], :kernel[1]], axis=(1, 2)) / (kernel[0] * kernel[1])
bin_signal = diff_frame_ratio > 2.75
bin_signal = binary_closing(bin_signal, structure=np.ones(40))
gaps = find_inactive(bin_signal)
# %%
model = models.load_model('src/resnet50v2conf_67.h5', backbone_name='resnet50')
labels_to_names = {0: 'abnormal'}


def detect(image):
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        # if score < 0.5:
        if score < 0.4:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


def cell_detect(frame):
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


# %%
v_max = 11
v_min = 2
maps = defaultdict(lambda: None)
for gap in gaps:
    lenght = gap[1] - gap[0]
    window = diff_frames[gap[0]:gap[0] + 50]
    sumframe = np.sum(window, axis=0)
    aveframe = (sumframe / len(window) + 1)
    aveframe[aveframe > v_max] = v_max
    aveframe[aveframe < v_min] = v_min
    normframe = (((aveframe - v_min) / (v_max - v_min)) * 255).astype('uint8')
    image = np.stack((normframe,) * 3, axis=-1)
    detect(image)
    # plt.imshow(normframe,vmax=255,vmin=0)
    # plt.show()
    # print("min:{} -- max:{}".format(aveframe.min(),aveframe.max()))
    # sumframe = cv2.normalize(sumframe, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype('uint8')
    # image = np.stack((normframe,) * 3, axis=-1)
    # cell_locs = cell_detect(normframe)
    # for i in range(gap[0], gap[1] + 1):
    #     o = {'area': None, 'cells': cell_locs}
    #     maps[i] = o
    # plt.imshow(aveframe,vmax=9,vmin=2.0)
    # plt.show()
    # plt.imshow(image)
    # plt.show()
    # detect(image)
    # plt.imshow(sumframe, cmap='grey')
    # plt.show()
    # n_vote = 5
    # step_size = int(lenght / (n_vote - 1))
    # window_size = 50
    # sum_size = window_size - 1
    # for last in range(gap[0] + sum_size, gap[1], step_size):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, last)
    #     _, frame = cap.read()
    #     area = find_count_area(frame)
# %%
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
trackers = cv2.MultiTracker_create()
tracker = cv2.TrackerCSRT_create()
frame_id = 0
prev_map = None
while True:
    ret, frame = cap.read()
    if frame is None:
        # There are no more frames available to stabilize
        break
    map = maps[frame_id]
    if map is not None:
        prev_map = None
        for cell in map['cells']:
            top, left, bottom, right = cell.bbox
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    else:
        if prev_map is None:
            trackers = cv2.MultiTracker_create()
            prev_map = maps[frame_id-1]
            for cell in prev_map['cells']:
                tracker = cv2.TrackerCSRT_create()
                trackers.add(tracker, frame, cell.bbox)
                # tracker.init(frame,cell.bbox)
        else:
            (success, boxes) = trackers.update(frame)
            if success:
                # top, left, bottom, right = boxes
                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                for top, left, bottom, right in boxes:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_id += 1
