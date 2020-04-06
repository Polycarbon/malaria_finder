from collections import OrderedDict

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QPolygonF
from scipy.spatial.distance import cdist
import numpy as np


class CellRect(QRectF):

    def __init__(self, *__args, score=None, isCount=False, count_id=None):
        QRectF.__init__(self, *__args)
        self.score = score
        self.__isCounted = isCount
        self.__count_id = count_id

    def getScore(self):
        return self.score

    def isCounted(self):
        return self.__isCounted

    def count(self, id):
        self.__isCounted = True
        self.__count_id = id

    def getCountId(self):
        assert self.__count_id is not None
        return self.__count_id

    def centroid(self):
        return self.center().x(), self.center().y()

    def translated(self, *__args):
        translate = super().translated(*__args)
        return CellRect(translate, score=self.score, isCount=self.__isCounted, count_id=self.__count_id)


class ObjectTracker():
    def __init__(self):
        self.__cells = OrderedDict()
        self.__disappeared = OrderedDict()
        self.nextObjectID = 0
        self.countId = 0
        self.distance_threshold = 80

    def __register(self, object):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.__cells[self.nextObjectID] = object
        self.__disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def __deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.__cells[objectID]
        del self.__disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        new_object = 0
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for object_id in list(self.__disappeared.keys()):
                self.__disappeared[object_id] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                # if self.disappeared[object_id] > self.maxDisappeared:
                #     self.deregister(object_id)

            # return early as there are no centroids or tracking info
            # to update
            return 0

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.__cells) == 0:
            new_object += len(rects)
            for i in range(0, len(rects)):
                self.__register(rects[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # initialize an array of input centroids for the current frame
            input_centroids = np.array([(o.center().x(), o.center().y()) for o in rects])
            # grab the set of object IDs and corresponding centroids
            object_ids = list(self.__cells.keys())
            object_centroids = np.array([(o.center().x(), o.center().y()) for o in self.__cells.values()])
            d = cdist(object_centroids, input_centroids)
            rows = d.min(axis=1).argsort()
            cols = d.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                if d[row, col] < self.distance_threshold:
                    rect = rects[col].getRect()
                    self.__cells[object_id].setRect(*rect)
                    self.__disappeared[object_id] = 0
                    used_rows.add(row)
                    used_cols.add(col)
            unused_rows = set(range(0, d.shape[0])).difference(used_rows)
            unused_cols = set(range(0, d.shape[1])).difference(used_cols)
            # if d.shape[0] >= d.shape[1]:
            # for row in unused_rows:
            #     object_id = object_ids[row]
            #     self.disappeared[object_id] += 1
            #     if self.disappeared[object_id] > self.maxDisappeared:
            #         self.deregister(object_id)
            # else:
            new_object += len(unused_cols)
            for col in unused_cols:
                print(rects[col])
                self.__register(rects[col])

        # return the set of trackable objects
        return new_object

    def translates(self, dx, dy):
        map(lambda o: o.translates(dx, dy), self.__cells.values())

    def translated(self, dx, dy):
        translated = OrderedDict([(k, cell.translated(dx, dy)) for k, cell in self.__cells.items()])
        self.__cells.update(translated)
        return translated

    def getObjects(self):
        return self.__cells.copy()

    def countInArea(self, area: QPolygonF):
        new_count = []
        for cell in self.__cells.values():
            if area.containsPoint(cell.center(), Qt.OddEvenFill) and not cell.isCounted():
                cell.count(self.countId)
                new_count.append(cell)
        return new_count
