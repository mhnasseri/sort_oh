import unittest

from libs.kalman_tracker import KalmanBoxTracker


class TestKalmanTracker(unittest.TestCase):
    bbox = [[1., 2., 3., 4.], [1.1, 2.1, 3.1, 4.1], [1.2, 2.2, 3.2, 4.2]]

    def test_ctor_increment_id(self):
        filter1 = KalmanBoxTracker(self.bbox[0], 0, self.bbox[0])
        filter2 = KalmanBoxTracker(self.bbox[0], 0, self.bbox[0])
        self.assertEqual(filter1.id + 1, filter2.id)

    def test_ctor_0(self):
        filter1 = KalmanBoxTracker(self.bbox[0], 0, self.bbox[0])
        self.assertListEqual(filter1.kf.x.tolist(), [[2.], [3.], [4.], [1.], [0.], [0.], [0.]])
        self.assertListEqual(filter1.get_state().tolist(), [self.bbox[0]])

    def test_ctor_1(self):
        filter1 = KalmanBoxTracker(self.bbox[0], 1, [1, 2.1, 3.2, 4.3])
        self.assertListEqual(filter1.kf.x.tolist(),
                             [[2.], [3.], [4.], [1.], [-0.10000000000000009], [-0.20000000000000018],
                              [-0.8399999999999999]])
        self.assertListEqual(filter1.get_state().tolist(), [self.bbox[0]])

    def test_0_predict(self):
        f = KalmanBoxTracker(self.bbox[0], 0, self.bbox[0])
        actual = f.predict()
        self.assertListEqual([self.bbox[0]], actual.tolist())
        actual = f.predict()
        self.assertListEqual([self.bbox[0]], actual.tolist())

    def test_1_predict(self):
        f = KalmanBoxTracker(self.bbox[0], 1, self.bbox[0])
        actual = f.predict()
        self.assertListEqual([self.bbox[0]], actual.tolist())
        actual = f.predict()
        self.assertListEqual([self.bbox[0]], actual.tolist())

    def test_0_update_0(self):
        f = KalmanBoxTracker(self.bbox[0], 0, self.bbox[0])
        f.update(self.bbox[0], 0)
        actual = f.predict()
        self.assertListEqual([self.bbox[0]], actual.tolist())

    def test_0_update_0_multiple(self):
        f = KalmanBoxTracker(self.bbox[0], 0, self.bbox[0])
        for x in self.bbox:
            f.update(x, 0)
        actual = f.predict()
        self.assertListEqual([self.bbox[0]], actual.tolist())

    def test_0_update_1_multiple(self):
        f = KalmanBoxTracker(self.bbox[0], 0, self.bbox[0])
        for x in self.bbox:
            f.update(x, 1)
        actual = f.predict()
        expected = [[1.096774193548387, 2.096774193548387, 3.096774193548387, 4.096774193548387]]
        self.assertListEqual(expected, actual.tolist())
        self.assertListEqual(expected, f.get_state().tolist())

    def test_1_update_1_negative(self):
        f = KalmanBoxTracker(self.bbox[0], 1, [1, 2, 3.5, 4.5])
        for x in [self.bbox[0], [1, 2, 1.5, 3.5], [1, 2, 2, 3], [1, 2, 1.1, 2.1]]:
            f.predict()
            f.update(x, 1)
        actual = f.predict()
        expected = [[0.7705099024374483, 1.8092846803759595, 0.9155530672670072, 1.9636854687429093]]
        self.assertListEqual(expected, actual.tolist())

    def test_0_update_1(self):
        f = KalmanBoxTracker(self.bbox[0], 0, self.bbox[0])
        f.update(self.bbox[0], 1)
        actual = f.predict()
        self.assertListEqual([self.bbox[0]], actual.tolist())


if __name__ == '__main__':
    unittest.main()
