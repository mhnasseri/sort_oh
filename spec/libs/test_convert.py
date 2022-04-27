import unittest
from libs import convert


class TestConvert(unittest.TestCase):
    def test_bbox_to_z(self):
        self.assertListEqual(convert.bbox_to_z([1, 2, 3, 4]).tolist(), [[2.0], [3.0], [4.0], [1.0]])

    def test_x_to_bbox(self):
        self.assertListEqual(convert.x_to_bbox([1, 2, 3, 4]).tolist(), [[-0.7320508075688772, 1.5669872981077806, 2.732050807568877, 2.433012701892219]])
        self.assertListEqual(convert.x_to_bbox([1, 2, 3, 4], 0.5).tolist(),
                             [[-0.7320508075688772, 1.5669872981077806, 2.732050807568877, 2.433012701892219, 0.5]])


if __name__ == '__main__':
    unittest.main()
