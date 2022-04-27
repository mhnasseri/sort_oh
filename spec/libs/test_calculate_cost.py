import unittest

import numpy as np

from libs import calculate_cost


class TestCalculateCost(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(calculate_cost.iou([0, 0, 1, 1], [0, 0, 1, 1]), 1)
        self.assertEqual(calculate_cost.iou([1, 1, 2, 3], [1, 1, 1.1, 1.2]), 0.010000000000000007)

    test_data_matrix = [
        {
            'detections': [
                [1, 1, 2, 3],
                [1, 2, 3, 4],
                [0, 0, 0.5, 0.2]
            ],
            'trackers': [
                [1, 1, 1.1, 1.2],
                [1, 3, 2, 5]
            ],
            'test': [
                {
                    'function': calculate_cost.cal_iou,
                    'expected': np.array([[0.009999999776482582, 0.0], [0.0, 0.20000000298023224], [0.0, 0.0]],
                                         np.float32)
                },
                {
                    'function': calculate_cost.cal_area_cost,
                    'expected': np.array([[99.0, 0.0], [199.0, 1.0], [4.0, 19.0]])
                },
            ]
        },
        {
            'detections': [
                [1, 1, 2, 3],
                [1, 2, 3, 4],
                [0, 0, 0.5, 0.2]
            ],
            'trackers': [],
            'test': [
                {
                    'function': calculate_cost.cal_iou,
                    'expected': np.array([[], [], []], np.float32)
                },
                {
                    'function': calculate_cost.cal_area_cost,
                    'expected': np.array([[], [], []], np.float32)
                }
            ]
        },
        {
            'detections': [
                [1, 1, 2, 3],
                [1, 2, 3, 4],
                [0, 0, 0.5, 0.2]
            ],
            'trackers': [1, 2, 3, 4],
            'test': [
                {
                    'function': calculate_cost.cal_ios,
                    'expected': np.array([[0.25], [1.0], [0.0]], np.float32)
                }
            ]
        }
    ]

    def test_cal_iou_area_cost_ios(self):
        for test_data in self.test_data_matrix:
            detections = test_data['detections']
            trackers = test_data['trackers']
            for test in test_data['test']:
                expected = test['expected']
                actual = test['function'](detections, trackers)
                self.assertListEqual(expected.tolist(), actual.tolist())

    def test_cal_outside(self):
        test_data_matrix = [
            {
                'trackers': [[-1, -200, 300, 400]],
                'img_s': [50, 50],
                'expected': np.array([[1.4138981]], np.float32)
            }
        ]

        for test in test_data_matrix:
            actual = calculate_cost.cal_outside(test['trackers'], test['img_s'])
            self.assertEqual(test['expected'], actual)


if __name__ == '__main__':
    unittest.main()
