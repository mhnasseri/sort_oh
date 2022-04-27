import numpy as np

from . import utils
from . import association
from . import kalman_tracker
from . import calculate_cost


def _init_area_and_trackers(kalman_trackers):
    """
    WARNING: This writes to kalman_trackers
    """
    trks = np.zeros((len(kalman_trackers), 5))
    to_del = []
    area_sum = 0
    for t, trk in enumerate(trks):
        pos = kalman_trackers[t].predict()[0]
        trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
        area_sum = area_sum + kalman_trackers[t].kf.x[2]

        if not np.isscalar(area_sum):
            area_sum = area_sum[0]
        if np.any(np.isnan(pos)):
            to_del.append(t)
    area_avg = 0
    if len(kalman_trackers) > 0:
        area_avg = area_sum / len(kalman_trackers)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
        kalman_trackers.pop(t)

    return trks, area_avg


def _remove_outside_trackers(trks, kalman_trackers, scene):
    """
    Warning: Writes to trks, kalman_trackers
    """
    outside = calculate_cost.cal_outside(trks, scene)
    to_del = []
    for t in range(len(outside)):
        if outside[t] > 0.5:
            to_del.append(t)
    for t in reversed(to_del):
        kalman_trackers.pop(t)
        trks = np.delete(trks, t, 0)

    return trks


def _update_matched_trackers(dets, kalman_trackers, unmatched_trks, occluded_trks, matched, trks):
    """
    update matched trackers with assigned detections
    """
    unmatched_trks_pos = []
    if len(dets) > 0:
        for t, trk in enumerate(kalman_trackers):
            if t not in unmatched_trks:
                if t not in occluded_trks:
                    # Update according to associated detection
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0], 1)
                else:
                    # Update according to estimated bounding box
                    trk.update(trks[t, :], 0)
            else:
                unmatched_trks_pos.append(np.concatenate((trk.get_state()[0], [trk.id + 1])).reshape(1, -1))

    return unmatched_trks_pos


def _build_new_targets(unmatched_before_before, unmatched_before, unmatched, area_avg, kalman_trackers):
    if (len(unmatched_before_before) != 0) and (len(unmatched_before) != 0) and (len(unmatched) != 0):
        new_trackers, del_ind = association.find_new_trackers_2(unmatched, unmatched_before,
                                                                unmatched_before_before, area_avg)
        if len(new_trackers) > 0:
            unm = np.asarray(unmatched)
            unmb = np.asarray(unmatched_before)
            unmbb = np.asarray(unmatched_before_before)
            # sort del_ind in descending order so removing step by step do not make error
            del_ind = np.sort(del_ind, axis=0)
            del_ind = np.flip(del_ind, axis=0)
            for i, new_tracker in enumerate(new_trackers):
                new_trk_certainty = unm[new_tracker[0], 4] + unmb[new_tracker[1], 4] + unmbb[new_tracker[2], 4]
                if new_trk_certainty > 2:
                    trk = kalman_tracker.KalmanBoxTracker(unm[new_tracker[0], :], 1, unmb[new_tracker[1], :])
                    kalman_trackers.append(trk)
                    # remove matched detection from unmatched arrays
                    unmatched.pop(del_ind[i, 0])
                    unmatched_before.pop(del_ind[i, 1])
                    unmatched_before_before.pop(del_ind[i, 2])


def _remove_dead_tracklet(kalman_trackers, max_age):
    ret = []
    i = len(kalman_trackers)
    for trk in reversed(kalman_trackers):
        d = trk.get_state()[0]
        if trk.time_since_update < 1:
            # +1 as MOT benchmark requires positive
            ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
        i -= 1
        # remove dead tracklet
        if trk.time_since_update > (np.minimum(7, max_age + kalman_trackers[i].age / 10)):
            kalman_trackers.pop(i)

    return ret


class Sort_OH(object):

    def __init__(self, max_age=3, min_hits=3, scene=np.array([1920, 1080])):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.area_avg_array = []
        self.frame_count = 0
        self.unmatched_before_before = []
        self.unmatched_before = []
        self.unmatched = []
        self.scene = scene
        self.conf_trgt = 0
        self.conf_objt = 0

    def to_json(self):
        """
        Returns a dict object that can be used to serialize this a JSON
        """
        return {
            "area_avg_array": list(map(lambda x: x if np.isscalar(x) else x[0], self.area_avg_array)),
            "conf_objt": self.conf_objt,
            "conf_trgt": self.conf_trgt,
            "frame_count": self.frame_count,
            "max_age": self.max_age,
            "min_hits": self.min_hits,
            "scene": self.scene.tolist(),
            "trackers": list(map(lambda x: x.to_json(), self.trackers)),
            "unmatched": list(map(lambda x: x.tolist(), self.unmatched)),
            "unmatched_before": list(map(lambda x: x.tolist(), self.unmatched_before)),
            "unmatched_before_before": list(map(lambda x: x.tolist(), self.unmatched_before_before)),
        }

    def update(self, dets, gts):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        trks, area_avg = _init_area_and_trackers(self.trackers)
        self.area_avg_array.append(area_avg)

        trks = _remove_outside_trackers(trks, self.trackers, self.scene)

        matched, unmatched_dets, unmatched_trks, occluded_trks, unmatched_gts = association.associate_detections_to_trackers(self, dets, trks, gts, area_avg)

        # update matched trackers with assigned detections
        unmatched_trks_pos = _update_matched_trackers(dets, self.trackers, unmatched_trks, occluded_trks, matched, trks)

        # create and initialise new trackers for unmatched detections
        if self.frame_count <= self.min_hits:
            for i in unmatched_dets:
                # Put condition on uncertainty
                if dets[i, 4] > 0.6:
                    # put dets[i, :] as dummy data for last argument
                    trk = kalman_tracker.KalmanBoxTracker(dets[i, :], 0, dets[i, :])
                    self.trackers.append(trk)
        else:
            self.unmatched = []
            for i in unmatched_dets:
                self.unmatched.append(dets[i, :])

            # Build new targets
            _build_new_targets(self.unmatched_before_before, self.unmatched_before, self.unmatched,
                               self.area_avg_array[len(self.area_avg_array) - 1], self.trackers)
            self.unmatched_before_before = self.unmatched_before
            self.unmatched_before = self.unmatched

        # get position of unmatched ground truths
        unmatched_gts_pos = []
        for g in unmatched_gts:
            unmatched_gts_pos.append(gts[g, :].reshape(1, 5))

        ret = _remove_dead_tracklet(self.trackers, self.max_age)

        out1 = np.empty((0, 5))
        out2 = np.empty((0, 5))
        out3 = np.empty((0, 5))

        if len(ret) > 0:
            out1 = np.concatenate(ret)
        if len(unmatched_trks_pos) > 0:
            out2 = np.concatenate(unmatched_trks_pos)
        if len(unmatched_gts_pos) > 0:
            out3 = np.concatenate(unmatched_gts_pos)

        return out1, out2, out3
