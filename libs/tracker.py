import numpy as np
from . import association
from . import kalman_tracker
from . import calculate_cost


class Sort_OH(object):

    def __init__(self, max_age=3, min_hits=3):
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
        self.seq = []
        self.conf_trgt = 0
        self.conf_objt = 0

    def update(self, dets, gts):
        """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        area_sum = 0
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            area_sum = area_sum + self.trackers[t].kf.x[2]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        area_avg = 0
        if len(self.trackers) > 0:
            area_avg = area_sum/len(self.trackers)
        self.area_avg_array.append(area_avg)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        # remove outside image trackers
        to_del = []
        if self.seq == 'MOT17-05-DPM' or self.seq == 'MOT17-05-FRCNN' or self.seq == 'MOT17-05-SDP' \
           or self.seq == 'MOT17-05-POI' or self.seq == 'MOT17-06-DPM' or self.seq == 'MOT17-06-FRCNN' \
           or self.seq == 'MOT17-06-SDP' or self.seq == 'MOT17-06-POI':
            scene = np.array([640, 480])
        else:
            scene = np.array([1920, 1080])
        outside = calculate_cost.cal_outside(trks, scene)
        for t in range(len(outside)):
            if outside[t] > 0.5:
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, 0)

        matched, unmatched_dets, unmatched_trks, occluded_trks, unmatched_gts = association.associate_detections_to_trackers(self, dets, trks, gts, area_avg)

        # update matched trackers with assigned detections
        unmatched_trks_pos = []
        if len(dets) > 0:
            for t, trk in enumerate(self.trackers):
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

        # create and initialise new trackers for unmatched detections
        if self.frame_count <= self.min_hits:
            for i in unmatched_dets:
                # Put condition on uncertainty
                if dets[i, 4] > 0.6:
                    trk = kalman_tracker.KalmanBoxTracker(dets[i, :], 0, dets[i, :]) # put dets[i, :] as dummy data for last argument
                    self.trackers.append(trk)
        else:
            self.unmatched = []
            for i in unmatched_dets:
                self.unmatched.append(dets[i, :])

            # Build new targets
            if (len(self.unmatched_before_before) != 0) and (len(self.unmatched_before) != 0) and (len(self.unmatched) != 0):
                new_trackers, del_ind = association.find_new_trackers_2(self.unmatched, self.unmatched_before, self.unmatched_before_before, area_avg)
                if len(new_trackers) > 0:
                    unm = np.asarray(self.unmatched)
                    unmb = np.asarray(self.unmatched_before)
                    unmbb = np.asarray(self.unmatched_before_before)
                    # sort del_ind in descending order so removing step by step do not make error
                    del_ind = np.sort(del_ind, axis=0)
                    del_ind = np.flip(del_ind, axis=0)
                    for i, new_tracker in enumerate(new_trackers):
                        new_trk_certainty = unm[new_tracker[0], 4] + unmb[new_tracker[1], 4] + unmbb[new_tracker[2], 4]
                        if new_trk_certainty > 2:
                            trk = kalman_tracker.KalmanBoxTracker(unm[new_tracker[0], :], 1, unmb[new_tracker[1], :])
                            self.trackers.append(trk)
                            # remove matched detection from unmatched arrays
                            self.unmatched.pop(del_ind[i, 0])
                            self.unmatched_before.pop(del_ind[i, 1])
                            self.unmatched_before_before.pop(del_ind[i, 2])
            self.unmatched_before_before = self.unmatched_before
            self.unmatched_before = self.unmatched

        # get position of unmatched ground truths
        unmatched_gts_pos = []
        for g in unmatched_gts:
            unmatched_gts_pos.append(gts[g, :].reshape(1, 5))

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1:
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > (np.minimum(7, self.max_age + self.trackers[i].age/10)):
                self.trackers.pop(i)
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
