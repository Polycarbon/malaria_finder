import cv2

global FRAME_COUNT, FPS, FRAME_WIDTH, FRAME_HEIGHT


def init(cap):
    global FRAME_COUNT, FPS, FRAME_WIDTH, FRAME_HEIGHT
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FPS))
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
