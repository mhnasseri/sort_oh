import numpy as np
from scipy.optimize import linear_sum_assignment

from . import utils
from . import calculate_cost
from . import visualization


def __array_of_ndarray_tolist(l):
    return utils.list_of_ndarray_tolist(l)


def __kalmantrackers_list_to_dict(kalman_trackers):
    return list(map(lambda x: {
        "time_since_observed": x.time_since_observed,
        "confidence": x.confidence if isinstance(x.confidence, float) or isinstance(x.confidence, int) else float(
            x.confidence.tolist()[0]),
        "age": x.age,
    }, kalman_trackers))


def __tracker_to_dict(tracker):
    return {
        "trackers": __kalmantrackers_list_to_dict(tracker.trackers),
        "frame_count": tracker.frame_count,
        "min_hits": tracker.min_hits,
        "conf_trgt": tracker.conf_trgt,
        "conf_objt": tracker.conf_objt,
        "unmatched_before": __array_of_ndarray_tolist(tracker.unmatched_before),
    }


def _unmatched_detections(detections, matched_indices):
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    return unmatched_detections


def _unmatched_trackers(trackers, matched_indices):
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    return unmatched_trackers


def _filter_low_iou(matched_indices, iou_matrix, iou_threshold, unmatched_detections, unmatched_trackers,
                    kalman_trackers):
    """
    WARNING: This writes to kalman_trackers, unmatched_detections, unmatched_trackers
    """
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            kalman_trackers[m[1]].time_since_observed = 0

    return matches


def _clean_up_to_delete(matches_ext_com, unmatched_detections, unmatched_trackers, matches, unmatched_before):
    """
    WARNING: This writes to unmatched_before
    """
    to_del_und = []
    to_del_unt = []
    to_del_undb = []
    matches_ext = np.empty((0, 2), dtype=int)
    if len(matches_ext_com) > 0:
        matches_ext_com = np.concatenate(matches_ext_com, axis=0)
        for m in matches_ext_com:
            new = np.array([[unmatched_detections[m[1]], unmatched_trackers[m[0]]]])
            matches_ext = np.append(matches_ext, new, axis=0)
            to_del_unt.append(m[0])
            to_del_und.append(m[1])
            to_del_undb.append(m[2])
    matches = np.concatenate((matches, matches_ext))
    if len(to_del_unt) > 0:
        to_del_unt = np.array(to_del_unt)
        to_del_unt = np.sort(to_del_unt)
        for i in reversed(to_del_unt):
            unmatched_trackers = np.delete(unmatched_trackers, i, 0)
    if len(to_del_und) > 0:
        to_del_und = np.array(to_del_und)
        to_del_und = np.sort(to_del_und)
        for i in reversed(to_del_und):
            unmatched_detections = np.delete(unmatched_detections, i, 0)
    if len(to_del_undb) > 0:
        to_del_undb = np.array(to_del_undb)
        to_del_undb = np.sort(to_del_undb)
        for i in reversed(to_del_undb):
            unmatched_before.pop(i)

    return matches, unmatched_trackers, unmatched_detections


def _occluded_trackers(frame_count, min_hits, ios_matrix, unmatched_trackers, trackers, kalman_trackers, average_area,
                       conf_trgt, conf_objt):
    """
    WARNING: This writes to kalman_trackers
    """
    occluded_trackers = []
    if frame_count > min_hits:
        trks_occlusion = np.amax(ios_matrix, axis=0)
        unm_trks = unmatched_trackers
        unmatched_trackers = []
        for ut in unm_trks:
            ut_area = (trackers[ut, 3] - trackers[ut, 1]) * (trackers[ut, 2] - trackers[ut, 0])
            kalman_trackers[ut].time_since_observed += 1
            kalman_trackers[ut].confidence = min(1, kalman_trackers[ut].age / (
                    kalman_trackers[ut].time_since_observed * 10) * (ut_area / average_area))
            if trks_occlusion[ut] > 0.3 and kalman_trackers[ut].confidence > conf_trgt:
                # if trks_occlusion[ut] > 0.3 and ut_area > 0.7 * average_area and mot_tracker.trackers[ut].age > 5:
                occluded_trackers.append(ut)
            elif kalman_trackers[ut].confidence > conf_objt:
                # elif mot_tracker.trackers[ut].age > (mot_tracker.trackers[ut].time_since_observed * 10 + 10) and mot_tracker.trackers[ut].time_since_observed < 5:
                occluded_trackers.append(ut)
            else:
                unmatched_trackers.append(ut)

    return occluded_trackers, unmatched_trackers


def _build_iou_matrix_ext(unmatched_trackers, unmatched_detections, detections, trackers, kalman_trackers):
    iou_matrix_ext = np.zeros((len(unmatched_trackers), len(unmatched_detections)), dtype=np.float32)
    for ud in range(len(unmatched_detections)):
        for ut in range(len(unmatched_trackers)):
            param_x = np.minimum(1.2, (kalman_trackers[unmatched_trackers[ut]].time_since_observed + 1) * 0.3)
            param_y = np.minimum(0.5, (kalman_trackers[unmatched_trackers[ut]].time_since_observed + 1) * 0.1)
            iou_matrix_ext[ut, ud] = calculate_cost.iou_ext_sep(detections[unmatched_detections[ud]],
                                                                trackers[unmatched_trackers[ut]],
                                                                param_x,
                                                                param_y)
    return iou_matrix_ext


def _filter_low_iou_low_area(matched_indices_ext, matched_indices, iou_matrix_ext, iou_threshold, iou_matrix):
    matches_ext_com = []
    del_ind = np.empty((0, 3), dtype=int)
    for m in matched_indices_ext:
        ind = matched_indices[np.where(matched_indices[:, 0] == m[1]), 1]
        if ind.size:
            if (iou_matrix_ext[m[0], m[1]] >= iou_threshold) and (iou_matrix[m[1], ind] >= iou_threshold):
                matches_ext_com.append(np.concatenate((m.reshape(1, 2), ind), axis=1))
                # remove matched detections from unmatched arrays
                del_ind = np.concatenate((del_ind, np.array([m[0], m[1], ind.item(0)]).reshape(1, 3)))

    return matches_ext_com, del_ind


def associate_detections_to_trackers(mot_tracker, detections, trackers, groundtruths, average_area,
                                     iou_threshold=0.3, ):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 5 lists of matches, unmatched_detections, unmatched_trackers, occluded_trackers and unmatched ground truths
    """
    if len(trackers) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), \
               np.arange(len(detections)), \
               np.empty((0, 1), dtype=int),\
               np.empty((0, 1), dtype=int),\
               np.empty((0, 1), dtype=int)

    # assign only according to iou
    # calculate intersection over union cost
    iou_matrix = calculate_cost.cal_iou(detections, trackers)
    ios_matrix = calculate_cost.cal_ios_matrix(trackers)

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.asarray(matched_indices)
    matched_indices = np.transpose(matched_indices)  # first column: detection indexes, second column: object indexes

    unmatched_detections = _unmatched_detections(detections, matched_indices)
    unmatched_trackers = _unmatched_trackers(trackers, matched_indices)
    matches = _filter_low_iou(matched_indices, iou_matrix, iou_threshold, unmatched_detections, unmatched_trackers,
                              mot_tracker.trackers)

    unmatched_detections = np.array(unmatched_detections)
    unmatched_trackers = np.array(unmatched_trackers)
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

        # try to match extended unmatched tracks to unmatched detections
        if len(unmatched_detections) > 0 and len(unmatched_trackers) > 0 and len(mot_tracker.unmatched_before) > 0:
            unm_dets = []
            for ud in unmatched_detections:
                unm_dets.append(detections[ud])

            iou_matrix = calculate_cost.cal_iou(unm_dets, mot_tracker.unmatched_before)
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.asarray(matched_indices)
            matched_indices = np.transpose(matched_indices)

            iou_matrix_ext = _build_iou_matrix_ext(unmatched_trackers, unmatched_detections, detections, trackers,
                                                   mot_tracker.trackers)
            matched_indices_ext = linear_sum_assignment(-iou_matrix_ext)
            matched_indices_ext = np.asarray(matched_indices_ext)
            matched_indices_ext = np.transpose(matched_indices_ext)

            matches_ext_com, del_ind = _filter_low_iou_low_area(matched_indices_ext, matched_indices, iou_matrix_ext,
                                                                iou_threshold, iou_matrix)

            matches, unmatched_trackers, unmatched_detections = _clean_up_to_delete(matches_ext_com,
                                                                                    unmatched_detections,
                                                                                    unmatched_trackers, matches,
                                                                                    mot_tracker.unmatched_before)

    occluded_trackers, unmatched_trackers = _occluded_trackers(mot_tracker.frame_count, mot_tracker.min_hits,
                                                               ios_matrix, unmatched_trackers, trackers,
                                                               mot_tracker.trackers, average_area,
                                                               mot_tracker.conf_trgt, mot_tracker.conf_objt)

    # find unmatched ground truths
    unmatched_groundtruths = []
    if visualization.DisplayState.display_gt_diff:
        found_trackers = trackers
        for i in reversed(np.sort(unmatched_trackers)):
            found_trackers = np.delete(found_trackers, i, 0)
        iou_matrix_1 = calculate_cost.cal_iou(groundtruths, found_trackers)
        # first column: ground truth indexes, second column: object indexes
        matched_indices_1 = linear_sum_assignment(-iou_matrix_1)
        matched_indices_1 = np.asarray(matched_indices_1)
        matched_indices_1 = np.transpose(matched_indices_1)

        for g, gt in enumerate(groundtruths):
            if g not in matched_indices_1[:, 0]:
                unmatched_groundtruths.append(g)

    return matches, unmatched_detections, unmatched_trackers, np.array(occluded_trackers), np.array(
        unmatched_groundtruths)


# def find_new_trackers(detections, detections_before, iou_threshold=0.3):
#     """
#   Detect new trackers from unmateched detection in current frame and before frame (both represented as bounding boxes)
#   Returns list of new trackers
#   """
#     if len(detections) == 0 or len(detections_before) == 0:
#         return np.empty((0, 2), dtype=int)
#     iou_matrix = calculate_cost.cal_iou(detections, detections_before)
#     # first column: detection indexes, second column: object indexes
#     matched_indices = linear_sum_assignment(-iou_matrix)
#     matched_indices = np.asarray(matched_indices)
#     matched_indices = np.transpose(matched_indices)
#
#     # filter out matched with low IOU
#     matches = []
#     for m in matched_indices:
#         if iou_matrix[m[0], m[1]] >= iou_threshold:
#             matches.append(m.reshape(1, 2))
#     if len(matches) == 0:
#         matches = np.empty((0, 2), dtype=int)
#     else:
#         matches = np.concatenate(matches, axis=0)
#
#     return matches


def find_new_trackers_2(detections, detections_before, detections_before_before, average_area, iou_threshold=0.3):
    """
    Detect new trackers from unmateched detection in current frame and before frame (both represented as bounding boxes)
    Returns list of new trackers
    """
    if len(detections) == 0 or len(detections_before) == 0 or len(detections_before_before) == 0:
        return np.empty((0, 2), dtype=int)
    iou_matrix = calculate_cost.cal_iou(detections, detections_before)
    iou_matrix_b = calculate_cost.cal_iou(detections_before, detections_before_before)

    # first column: detections indexes, second column: detections_before indexes
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.asarray(matched_indices)
    matched_indices = np.transpose(matched_indices)

    # first column: detections_before indexes, second column: detections_before_before indexes
    matched_indices_b = linear_sum_assignment(-iou_matrix_b)
    matched_indices_b = np.asarray(matched_indices_b)
    matched_indices_b = np.transpose(matched_indices_b)

    # filter out matched with low IOU and low area
    matches = []
    del_ind = np.empty((0, 3), dtype=int)
    for m in matched_indices:
        ind = matched_indices_b[np.where(matched_indices_b[:, 0] == m[1]), 1]
        if ind.size:
            if (iou_matrix[m[0], m[1]] >= iou_threshold) and (iou_matrix_b[m[1], ind] >= iou_threshold):
                matches.append(np.concatenate((m.reshape(1, 2), ind), axis=1))
                del_ind = np.concatenate((del_ind, np.array([m[0], m[1], ind.item(0)]).reshape(1, 3)))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, del_ind
