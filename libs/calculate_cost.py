from numba import jit
import numpy as np


@jit
def iou(bb_det, bb_trk):
    """
  Computes IOU (Intersection Over Union) between two bounding boxes in the form [x1,y1,x2,y2]
  """
    xx1 = np.maximum(bb_det[0], bb_trk[0])
    xx2 = np.minimum(bb_det[2], bb_trk[2])
    w = np.maximum(0., xx2 - xx1)
    if w == 0:
        return 0
    yy1 = np.maximum(bb_det[1], bb_trk[1])
    yy2 = np.minimum(bb_det[3], bb_trk[3])
    h = np.maximum(0., yy2 - yy1)
    if h == 0:
        return 0
    wh = w * h
    area_det = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
    area_trk = (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1])
    o = wh / (area_det + area_trk - wh)
    return o


#@jit
def iou_ext(bb_det, bb_trk, ext):
    """
    Computes extended IOU (Intersection Over Union) between two bounding boxes in the form [x1,y1,x2,y2]
    """
    trk_w = bb_trk[2] - bb_trk[0]
    trk_h = bb_trk[3] - bb_trk[1]
    xx1 = np.maximum(bb_det[0], bb_trk[0] - trk_w*ext/2)
    xx2 = np.minimum(bb_det[2], bb_trk[2] + trk_w*ext/2)
    w = np.maximum(0., xx2 - xx1)
    if w == 0:
        return 0
    yy1 = np.maximum(bb_det[1], bb_trk[1] - trk_h*ext/2)
    yy2 = np.minimum(bb_det[3], bb_trk[3] + trk_h*ext/2)
    h = np.maximum(0., yy2 - yy1)
    if h == 0:
        return 0
    wh = w * h
    area_det = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
    area_trk = (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1])
    o = wh / (area_det + area_trk - wh)
    return o


def iou_ext_sep(bb_det, bb_trk, ext_w, ext_h):
    """
    Computes extended IOU (Intersection Over Union) between two bounding boxes in the form [x1,y1,x2,y2]
    with separate extension coefficient
    """
    trk_w = bb_trk[2] - bb_trk[0]
    trk_h = bb_trk[3] - bb_trk[1]
    xx1 = np.maximum(bb_det[0], bb_trk[0] - trk_w*ext_w/2)
    xx2 = np.minimum(bb_det[2], bb_trk[2] + trk_w*ext_w/2)
    w = np.maximum(0., xx2 - xx1)
    if w == 0:
        return 0
    yy1 = np.maximum(bb_det[1], bb_trk[1] - trk_h*ext_h/2)
    yy2 = np.minimum(bb_det[3], bb_trk[3] + trk_h*ext_h/2)
    h = np.maximum(0., yy2 - yy1)
    if h == 0:
        return 0
    wh = w * h
    area_det = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
    area_trk = (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1])
    o = wh / (area_det + area_trk - wh)
    return o


@jit
def outside(trk, img_s):
    """
    Computes how many percent of trk is placed outside of img_s
    """
    out_x = 0
    out_y = 0
    if trk[0] < 0:
        out_x = -trk[0]
    if trk[2] > img_s[0]:
        out_x = trk[2] - img_s[0]
    if trk[1] < 0:
        out_y = -trk[1]
    if trk[3] > img_s[1]:
        out_y = trk[3] - img_s[1]
    out_a = out_x * (trk[3] - trk[1]) + out_y * (trk[2] - trk[0])
    area = (trk[3] - trk[1]) * (trk[2] - trk[0])
    return out_a / area


@jit
def area_cost(bb_det, bb_trk):
    """
    This cost compute the difference between bounding box sizes
    If 2 bounding boxes have the same size, then the ratio of their
    width and height is 1. So, this ideal case should have zero cost
    If we divide width and height of first to second bounding box, the
    values may become bigger or smaller than 1. To avoid this, the bigger
    number is divided to smaller number.
    Also, the width and height is considered separately, to consider
    aspect ratio. If only area is considered, 2 bounding box may had same
    area, but different width and height and this situation is considered
    best, but it is not.
    """
    w_d = bb_det[2] - bb_det[0]
    w_t = bb_trk[2] - bb_trk[0]
    h_d = bb_det[3] - bb_det[1]
    h_t = bb_trk[3] - bb_trk[1]
    if w_d > w_t:
        w_ratio = w_d / w_t
    else:
        w_ratio = w_t / w_d

    if h_d > h_t:
        h_ratio = h_d / h_t
    else:
        h_ratio = h_t / h_d

    ratio = w_ratio * h_ratio
    return ratio - 1


@jit
# intersection over union with limit on area
def iou_la(bb_det, bb_trk):
    """
  Computes IOU (Intersection Over Union) between two bounding boxes in the form [x1,y1,x2,y2]
  with limitation on percentage of change of area
  """
    xx1 = np.maximum(bb_det[0], bb_trk[0])
    xx2 = np.minimum(bb_det[2], bb_trk[2])
    w = np.maximum(0., xx2 - xx1)
    if w == 0:
        return 0
    yy1 = np.maximum(bb_det[1], bb_trk[1])
    yy2 = np.minimum(bb_det[3], bb_trk[3])
    h = np.maximum(0., yy2 - yy1)
    if h == 0:
        return 0
    wh = w * h
    area_det = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
    area_trk = (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1])
    o = wh / (area_det + area_trk - wh)

    # if detection area is very different from tracker area, then return zero as intersection
    area_ratio = area_det / area_trk
    if area_ratio > 1.45 or area_ratio < 0.6:
        o = 0
    return o


@jit
def ios(bb_first, bb_second):
    """
  Computes IOS (Intersection Over Second Bounding Box) between two bounding boxes in the form [x1,y1,x2,y2]
  """
    xx1 = np.maximum(bb_first[0], bb_second[0])
    yy1 = np.maximum(bb_first[1], bb_second[1])
    xx2 = np.minimum(bb_first[2], bb_second[2])
    yy2 = np.minimum(bb_first[3], bb_second[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_second[2] - bb_second[0]) * (bb_second[3] - bb_second[1]))
    return o


def cal_iou(detections, trackers):
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    return iou_matrix


def cal_area_cost(detections, trackers):
    if len(detections) == 0 or len(trackers) == 0:
        return np.empty((len(detections), len(trackers)), dtype=int)
    area_cost_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            area_cost_matrix[d, t] = area_cost(det, trk)
    return area_cost_matrix


def cal_ios(detections, tracker):
    ios_matrix = np.zeros((len(detections), 1), dtype=np.float32)
    for d, det in enumerate(detections):
        ios_matrix[d, 0] = ios(det, tracker)
    return ios_matrix


def cal_outside(trackers, img_s):
    out = np.zeros(len(trackers), dtype=np.float32)
    for t, trk in enumerate(trackers):
        out[t] = outside(trk, img_s)
    return out


def cal_ios_matrix(trackers):
    ios_matrix = np.zeros((len(trackers), len(trackers)), dtype=np.float32)
    for t1, trk1 in enumerate(trackers):
        for t2, trk2 in enumerate(trackers):
            if t2 != t1:
                ios_matrix[t1, t2] = ios(trk1, trk2)
    return ios_matrix


