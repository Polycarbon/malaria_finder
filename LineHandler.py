import math

import cv2
import numpy as np
from scipy.spatial import distance


def get_lines(lines_in):
    if cv2.__version__ < '3.0':
        return lines_in[0]
    return [l[0] for l in lines_in]


def merge_lines_pipeline_2(lines):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 3
    min_angle_to_merge = 4

    for line in lines:
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                    orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                    if int(abs(
                            abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                        # print("angles", orientation_i, orientation_j)
                        # print(int(abs(orientation_i - orientation_j)))
                        group.append(line)

                        create_new_group = False
                        group_updated = True
                        break

            if group_updated:
                break

        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                    orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                    if int(abs(
                            abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                        # print("angles", orientation_i, orientation_j)
                        # print(int(abs(orientation_i - orientation_j)))

                        new_group.append(line2)

                        # remove line from lines list
                        # lines[idx] = False
            # append new group
            super_lines.append(new_group)

    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))

    return super_lines_final


def merge_lines_segments1(lines, use_log=False):
    if (len(lines) == 1):
        return lines[0]

    line_i = lines[0]

    # orientation
    orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))

    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])

    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):

        # sort by y
        points = sorted(points, key=lambda point: point[1])

        if use_log:
            print("use y")
    else:

        # sort by x
        points = sorted(points, key=lambda point: point[0])

        if use_log:
            print("use x")

    return [points[0], points[len(points) - 1]]


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
# https://stackoverflow.com/questions/32702075/what-would-be-the-fastest-way-to-find-the-maximum-of-all-possible-distances-betw
def lines_close(line1, line2):
    dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
    dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
    dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
    dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])

    if (min(dist1, dist2, dist3, dist4) < 100):
        return True
    else:
        return False


def lineMagnitude(x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude


# Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
# https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
# http://paulbourke.net/geometry/pointlineplane/
def DistancePointLine(px, py, x1, y1, x2, y2):
    # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        # // closest point does not fall within the line segment, take the shorter distance
        # // to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine


def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1],
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1],
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1],
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1],
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])

    return min(dist1, dist2, dist3, dist4)


def extractLines(gray, threshold=0.7):
    binary = gray > threshold * 255
    edges = (binary * 255).astype('uint8')
    min_line_length = int(min(gray.shape) / 2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=min_line_length, maxLineGap=30)

    # ------------------
    # prepare
    _lines = []
    for _line in get_lines(lines):
        _lines.append([(_line[0], _line[1]), (_line[2], _line[3])])

    # sort
    _lines_x = []
    _lines_y = []
    for line_i in _lines:
        orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))
        if (abs(math.degrees(orientation_i)) > 80) and abs(math.degrees(orientation_i)) < (90 + 10):
            _lines_y.append(line_i)
        elif (abs(math.degrees(orientation_i)) < 10) or abs(math.degrees(orientation_i)) > (90 + 80):
            _lines_x.append(line_i)

    _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
    _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

    horizontals = merge_lines_pipeline_2(_lines_x)
    verticals = merge_lines_pipeline_2(_lines_y)

    return verticals, horizontals


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)


def segmented_intersections(verticals, horizontals):
    """Finds the intersections between groups of lines."""

    intersections = []
    for line1 in verticals:
        for line2 in horizontals:
            intersections.append(intersection(line1, line2))

    return intersections


def extend_verticals(verticals, x_bound, y_bound, add_bounding=True):
    new_lines = []
    for line in verticals:
        p1, p2 = line
        coef = np.polyfit([p1[1], p2[1]], [p1[0], p2[0]], 1)
        polynomial = np.poly1d(coef)
        x = polynomial(y_bound)
        p3 = (int(x[0]), y_bound[0])
        p4 = (int(x[1]), y_bound[1])
        new_lines.append([p3, p4])
    if add_bounding:
        new_lines.append([(x_bound[0], y_bound[0]), (x_bound[0], y_bound[1])])
        new_lines.append([(x_bound[1], y_bound[0]), (x_bound[1], y_bound[1])])
    return new_lines


def extend_horizontals(horizontals, x_bound, y_bound, add_bounding=True):
    new_lines = []
    for line in horizontals:
        p1, p2 = line
        coef = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)
        polynomial = np.poly1d(coef)
        y = polynomial(x_bound)
        p3 = (x_bound[0], int(y[0]))
        p4 = (x_bound[1], int(y[1]))
        new_lines.append([p3, p4])
    if add_bounding:
        new_lines.append([(x_bound[0], y_bound[0]), (x_bound[1], y_bound[0])])
        new_lines.append([(x_bound[0], y_bound[1]), (x_bound[1], y_bound[1])])
    return new_lines


def calculateBoundingPoints(pt, verticals, horizontals):
    intersects = segmented_intersections(verticals, horizontals)
    intersects = sorted(intersects, key=lambda p: distance.euclidean(p, pt))
    intersects = np.array(intersects,dtype=np.int)
    topleft = intersects[np.logical_and(intersects[:, 0] < pt[0], intersects[:, 1] < pt[1])][0]
    topright = intersects[np.logical_and(intersects[:, 0] > pt[0], intersects[:, 1] < pt[1])][0]
    bottomleft = intersects[np.logical_and(intersects[:, 0] < pt[0], intersects[:, 1] > pt[1])][0]
    bottomright = intersects[np.logical_and(intersects[:, 0] > pt[0], intersects[:, 1] > pt[1])][0]
    return [topleft, topright, bottomright, bottomleft, topleft]
