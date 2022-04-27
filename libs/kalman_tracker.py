import numpy as np
from filterpy.kalman import KalmanFilter
from . import convert


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, init_mode, bbox_before):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # (u, v, s, r, u_dot, v_dot, s_dot) -> (u,v): location center, s: area, r: aspect ratio
        # dim_x: Number of state variables for the Kalman filter
        # dim_z: Number of of measurement inputs a.k.a observations
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # state transition matrix
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]])
        # measurement function a.k.a stateProjection
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]])

        if init_mode == 0:
            # Measurement noise matrix
            self.kf.R[2:, 2:] *= 1.   # 10.

            # covariance matrix
            self.kf.P[4:, 4:] *= 10.  # 1000. # give high uncertainty to the unobservable initial velocities
            self.kf.P *= 10.

            # Process uncertainty/noise
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01

            # filter state estimate
            self.kf.x[:4] = convert.bbox_to_z(bbox)

        elif init_mode == 1:
            # Measurement noise matrix
            self.kf.R[2:, 2:] *= 1.

            # covariance matrix
            self.kf.P[4:, 4:] *= 10.  # give high uncertainty to the unobservable initial velocities
            # self.kf.P *= 10.

            # Process uncertainty/noise
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01

            state_before = convert.bbox_to_z(bbox_before)
            state = convert.bbox_to_z(bbox)
            # filter state estimate
            self.kf.x[:4] = state
            self.kf.x[4:] = state[0:3] - state_before[0:3]

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.age = 0
        # self.oc_number = 0  # Number of time an object is occluded
        self.time_since_observed = 0    # The period that an object is detected as occluded
        self.confidence = 0.5

    def update(self, bbox, isz):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        if isz == 0:
            # decrease area change ratio
            self.kf.x[6] /= 2
            self.kf.update(None)
        elif isz == 1:
            self.kf.update(convert.bbox_to_z(bbox))
            self.time_since_observed = 0

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # to prevent area become negative after prediction, make zero the rate of area change
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

        return convert.x_to_bbox(self.kf.x)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert.x_to_bbox(self.kf.x)

    def to_json(self):
        return {
            "time_since_observed": self.time_since_observed,
            "confidence": self.confidence if isinstance(self.confidence, float) or isinstance(self.confidence, int) else float(self.confidence.tolist()[0]),
            "age": self.age,
            "get_state": self.get_state().tolist(),
            "time_since_update": self.time_since_update,
            "id": self.id,
            "kf": {
                "x": self.kf.x.tolist(),
                "covariance": self.kf.P.tolist()
            }
        }
