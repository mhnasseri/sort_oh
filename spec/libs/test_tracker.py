import json
import unittest

import numpy as np

from libs.convert import x_to_bbox
from libs.kalman_tracker import KalmanBoxTracker
from libs.tracker import Sort_OH, _remove_outside_trackers, _build_new_targets, _remove_dead_tracklet, \
    _init_area_and_trackers, _update_matched_trackers
from libs.utils import DictObj, list_of_ndarray_tolist


def kalman_box_tracker_to_json_no_state(t):
    tjson = t
    if isinstance(t, KalmanBoxTracker):
        tjson = t.to_json()

    tjson.pop('get_state')
    tjson.pop('time_since_update')
    tjson.pop('id')
    tjson.pop('kf')
    return tjson


def kalman_tracker_mock_from_json(t):
    t_obj = DictObj(t)
    t_obj.update = lambda x, y: None

    if 'get_state' in t.keys():
        t_obj.get_state = lambda: t['get_state']
    if 'kf' in t.keys():
        t_obj.predict = lambda: x_to_bbox(t['kf']['x'])

    return t_obj

class TestTracker(unittest.TestCase):
    def assertNdArrayEqual(self, actual, expected, index=None):
        if index is None:
            index_copy = []
        else:
            index_copy = index.copy()

        self.assertEqual(len(actual), len(expected), msg="Length at index %s" % index_copy)
        for i in range(len(expected)):
            index_copy.append(i)

            if isinstance(expected[i], list):
                self.assertNdArrayEqual(actual[i], expected[i], index=index_copy)
            else:
                self.assertAlmostEqual(actual[i], expected[i], msg="Value at index %s" % index_copy)

    def assertSortOHJsonDictEqual(self, actual, expected):
        self.assertEqual(actual['max_age'], expected['max_age'])
        self.assertEqual(actual['min_hits'], expected['min_hits'])
        self.assertListEqual(
            list(map(lambda t: kalman_box_tracker_to_json_no_state(t), actual['trackers'])),
            expected['trackers'])
        self.assertEqual(len(actual['area_avg_array']), len(expected['area_avg_array']))
        for i in range(len(actual['area_avg_array'])):
            self.assertAlmostEqual(actual['area_avg_array'][i], expected['area_avg_array'][i], msg="index: %i" % i)
        self.assertEqual(actual['frame_count'], expected['frame_count'])
        self.assertNdArrayEqual(actual['unmatched_before_before'], expected['unmatched_before_before'])
        self.assertNdArrayEqual(actual['unmatched_before'], expected['unmatched_before'])
        self.assertNdArrayEqual(actual['unmatched'], expected['unmatched'])
        self.assertNdArrayEqual(actual['scene'], expected['scene'])
        self.assertEqual(actual['conf_trgt'], expected['conf_trgt'])
        self.assertEqual(actual['conf_objt'], expected['conf_objt'])

    def test_to_json_empty(self):
        self.assertSortOHJsonDictEqual(Sort_OH().to_json(), {
            "max_age": 3,
            "min_hits": 3,
            "trackers": [],
            "area_avg_array": [],
            "frame_count": 0,
            "unmatched_before_before": [],
            "unmatched_before": [],
            "unmatched": [],
            "scene": [1920, 1080],
            "conf_trgt": 0,
            "conf_objt": 0
        })
        self.assertSortOHJsonDictEqual(Sort_OH(1, 2).to_json(), {
            "max_age": 1,
            "min_hits": 2,
            "trackers": [],
            "area_avg_array": [],
            "frame_count": 0,
            "unmatched_before_before": [],
            "unmatched_before": [],
            "unmatched": [],
            "scene": [1920, 1080],
            "conf_trgt": 0,
            "conf_objt": 0
        })

    def test_to_json_filled(self):
        detections = [[1359.1, 413.27, 1479.36, 776.04, 2.4731], [584.04, 446.86, 668.7819999999999, 703.09, 1.2369],
                      [729.0, 457.0, 768.0, 576.0, 0.40858]]
        groundtruths = [[912.0, 484.0, 1009.0, 593.0, 0.0], [1342.0, 417.0, 1510.0, 797.0, 1.0],
                        [586.0, 446.0, 671.0, 710.0, 1.0], [1585.0, -1.0, 1921.0, 577.0, 0.0],
                        [1163.0, 441.0, 1196.0, 530.0, 0.0], [1308.0, 431.0, 1342.0, 549.0, 0.0],
                        [1422.0, 431.0, 1605.0, 768.0, 1.0], [1055.0, 483.0, 1091.0, 593.0, 1.0],
                        [1090.0, 484.0, 1122.0, 598.0, 1.0], [733.0, 487.0, 763.0, 555.0, 0.0],
                        [679.0, 492.0, 731.0, 597.0, 0.0], [737.0, 457.0, 764.0, 532.0, 0.0],
                        [1255.0, 447.0, 1288.0, 547.0, 1.0], [1015.0, 430.0, 1055.0, 546.0, 1.0],
                        [1100.0, 440.0, 1138.0, 548.0, 1.0], [934.0, 435.0, 976.0, 549.0, 1.0],
                        [442.0, 446.0, 549.0, 728.0, 1.0], [636.0, 458.0, 697.0, 645.0, 1.0],
                        [1365.0, 434.0, 1417.0, 558.0, 1.0], [1480.0, 433.0, 1542.0, 558.0, 1.0],
                        [473.0, 460.0, 562.0, 709.0, 1.0], [835.0, 473.0, 887.0, 548.0, 0.0],
                        [796.0, 476.0, 851.0, 536.0, 0.0], [547.0, 464.0, 582.0, 557.0, 1.0],
                        [375.0, 446.0, 416.0, 550.0, 0.0], [418.0, 459.0, 458.0, 543.0, 1.0],
                        [582.0, 455.0, 616.0, 589.0, 1.0], [972.0, 456.0, 1004.0, 533.0, 1.0],
                        [693.0, 462.0, 714.0, 529.0, 0.0], [712.0, 477.0, 732.0, 534.0, 0.0],
                        [733.0, 504.0, 764.0, 549.0, 0.0], [910.0, 408.0, 936.0, 537.0, 0.0],
                        [730.0, 509.0, 767.0, 569.0, 0.0], [679.0, 528.0, 725.0, 607.0, 0.0],
                        [1003.0, 453.0, 1021.0, 514.0, 0.0], [578.0, 431.0, 598.0, 474.0, 1.0],
                        [595.0, 428.0, 613.0, 470.0, 1.0], [1035.0, 452.0, 1060.0, 519.0, 1.0],
                        [664.0, 451.0, 698.0, 536.0, 1.0]]

        t = Sort_OH()
        for i in range(5):
            t.update(np.array(detections), np.array(groundtruths))

        self.assertSortOHJsonDictEqual(t.to_json(), {
            "max_age": 3,
            "min_hits": 3,
            "trackers": [
                {'age': 4, 'confidence': 0.5, 'time_since_observed': 0},
                {'age': 4, 'confidence': 0.5, 'time_since_observed': 0}
            ],
            "area_avg_array": [0, 32670.08143, 32670.08143, 32670.08143, 32670.08143],
            "frame_count": 5,
            "unmatched_before_before": [[7.2900e+02, 4.5700e+02, 7.6800e+02, 5.7600e+02, 4.0858e-01]],
            "unmatched_before": [[7.2900e+02, 4.5700e+02, 7.6800e+02, 5.7600e+02, 4.0858e-01]],
            "unmatched": [[7.2900e+02, 4.5700e+02, 7.6800e+02, 5.7600e+02, 4.0858e-01]],
            "scene": [1920, 1080],
            "conf_trgt": 0,
            "conf_objt": 0
        })

    def test_update_invocations(self):
        with open('spec/res/tracker_update.json') as f:
            invocations = json.load(f)['invocations']

            # Reset Kalman Tracker ID which got incremented in previous tests
            KalmanBoxTracker.count = 0

            # Make sure tracker is built as in the invocations
            t = Sort_OH(invocations[0]['self']['max_age'], invocations[0]['self']['min_hits'], np.array([1920, 1080]))
            t.conf_trgt = invocations[0]['self']['conf_trgt']
            t.conf_objt = invocations[0]['self']['conf_objt']

            for x in invocations:
                in_dets = np.array(x['dets'])
                in_gts = np.array(x['gts'])
                trackers, unmatched_trks_pos, unmatched_gts_pos = t.update(in_dets, in_gts)

                # Check main outputs
                self.assertNdArrayEqual(trackers, x['ret_ret'])
                self.assertNdArrayEqual(unmatched_trks_pos, x['ret_unmatched_trks_pos'])
                self.assertNdArrayEqual(unmatched_gts_pos, x['ret_unmatched_gts_pos'])

                # Check for side effects on inputs
                self.assertNdArrayEqual(in_dets, x['ret_dets'])
                self.assertNdArrayEqual(in_gts, x['ret_gts'])
                x['ret_self']['scene'] = [1920, 1080]  # HACK: Supply this info after refactoring manually
                self.assertSortOHJsonDictEqual(t.to_json(), x['ret_self'])

    def test__init_area_and_trackers(self):
        x = {
            "kalman_trackers": [
                {"time_since_observed": 0, "confidence": 0.5, "age": 100, "get_state": [[652.5021638258046, 429.89314515922297, 752.8235272240348, 732.8593911610949]], "time_since_update": 1, "id": 1, "kf": {"x": [702.6628455249197, 581.3762681601589, 30393.986862551403, 0.3311304962916904, 1.8372252619403222, -0.015659604650060353, 48.40383331102944]}},
                {"time_since_observed": 0, "confidence": 1, "age": 80, "get_state": [[507.46718038184713, 444.4740225602785, 612.0308138979817, 760.1653184821718]], "time_since_update": 1, "id": 4, "kf": {"x": [559.7489971399144, np.nan, 33009.82897101042, 0.33122114821311116, -0.13915475487793044, 0.08126716203647405, 7.688792975854]}},
                {"time_since_observed": 0, "confidence": 0.5, "age": 18, "get_state": [[759.1778340001522, 447.9602113881072, 843.5349907598177, 703.0550763755507]], "time_since_update": 1, "id": 6, "kf": {"x": [801.3564123799849, 575.5076438818289, 21519.077514331446, 0.33068935654121345, 2.916506192292385, 0.6870007981125773, 212.2750733010161]}},
                {"time_since_observed": 0, "confidence": 0.4204631042226629, "age": 16, "get_state": [[559.8197969819889, 437.24035029259085, 678.86542161119, 796.3669335084915]], "time_since_update": 1, "id": 7, "kf": {"x": [619.3426092965894, 616.8036419005412, 42752.44841988766, 0.33148652924318034, -0.06572298440042505, 0.32076230568408565, -52.95193989110717]}}
            ],
            "ret_kalman_trackers": [
                {"time_since_observed": 0, "confidence": 0.5, "age": 100, "get_state": [[652.5021638258046, 429.89314515922297, 752.8235272240348, 732.8593911610949]], "time_since_update": 1, "id": 1, "kf": {"x": [702.6628455249197, 581.3762681601589, 30393.986862551403, 0.3311304962916904, 1.8372252619403222, -0.015659604650060353, 48.40383331102944]}},
                {"time_since_observed": 0, "confidence": 0.5, "age": 18, "get_state": [[759.1778340001522, 447.9602113881072, 843.5349907598177, 703.0550763755507]], "time_since_update": 1, "id": 6, "kf": {"x": [801.3564123799849, 575.5076438818289, 21519.077514331446, 0.33068935654121345, 2.916506192292385, 0.6870007981125773, 212.2750733010161]}},
                {"time_since_observed": 0, "confidence": 0.4204631042226629, "age": 16, "get_state": [[559.8197969819889, 437.24035029259085, 678.86542161119, 796.3669335084915]], "time_since_update": 1, "id": 7, "kf": {"x": [619.3426092965894, 616.8036419005412, 42752.44841988766, 0.33148652924318034, -0.06572298440042505, 0.32076230568408565, -52.95193989110717]}}
            ],
            "ret_trks": [
                [652.5021638258046, 429.89314515922297, 752.8235272240348, 732.8593911610949, 0.0],
                [759.1778340001522, 447.9602113881072, 843.5349907598177, 703.0550763755507, 0.0],
                [559.8197969819889, 437.24035029259085, 678.86542161119, 796.3669335084915, 0.0]
            ],
            "ret_area_avg": 31918.83544194523
        }
        kalman_trackers = list(map(lambda t: kalman_tracker_mock_from_json(t), x['kalman_trackers']))
        trks, area_avg = _init_area_and_trackers(kalman_trackers)
        self.assertListEqual(trks.tolist(), x["ret_trks"])
        self.assertEqual(area_avg, x["ret_area_avg"])
        self.maxDiff = None
        self.assertEqual(len(kalman_trackers), len(x['ret_kalman_trackers']))
        self.assertListEqual(
            list(map(lambda t: kalman_tracker_mock_from_json(t) if isinstance(t, KalmanBoxTracker) else t, kalman_trackers)),
            list(map(lambda t: kalman_tracker_mock_from_json(t), x['ret_kalman_trackers'])))

    def test__remove_outside_trackers(self):
        x = {"trks": [[1359.1, 1413.27, 1479.3600000000001, 1776.04, 0.0],
                      [584.04, 446.86, 668.7819999999999, 703.09, 0.0]], "scene": [1920, 1080],
             "kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 1},
                                 {"time_since_observed": 0, "confidence": 0.5, "age": 1}],
             "ret_kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 1}],
             "ret_trks": [[584.04, 446.86, 668.7819999999999, 703.09, 0.0]]}
        kalman_trackers = list(map(lambda t: DictObj(t), x['kalman_trackers']))
        actual = _remove_outside_trackers(x['trks'], kalman_trackers, np.array(x['scene']))
        self.assertListEqual(actual.tolist(), x['ret_trks'])
        self.assertListEqual(kalman_trackers, list(map(lambda t: DictObj(t), x['ret_kalman_trackers'])))

    def test__update_matched_trackers(self):
        def testAndExpect(x):
            kalman_trackers = list(map(lambda t: kalman_tracker_mock_from_json(t), x['kalman_trackers']))
            trks = np.array(x['trks'])
            unmatched_trks_pos = _update_matched_trackers(np.array(x['dets']), kalman_trackers, x['unmatched_trks'],
                                                          x['occluded_trks'], np.array(x['matched']), trks)
            self.assertListEqual(
                list(map(lambda x: x.tolist(), unmatched_trks_pos[0])),
                x['ret_unmatched_trks_pos']
            )
            self.assertListEqual(
                list(map(lambda x: x.tolist(), trks)),
                x['ret_trks']
            )
            self.assertListEqual(
                list(map(lambda t: kalman_tracker_mock_from_json(t) if isinstance(t, KalmanBoxTracker) else t,
                         kalman_trackers)),
                list(map(lambda t: kalman_tracker_mock_from_json(t), x['ret_kalman_trackers'])))

        testAndExpect({
            "dets": [
                [1434.1, 389.14, 1582.3899999999999, 836.0, 2.3735],
                [589.13, 442.1, 680.026, 716.79, 1.3351],
                [1528.8, 413.27, 1649.06, 776.04, 1.3038],
                [729.0, 449.0, 768.0, 568.0, 0.55511],
                [1254.6, 446.72, 1288.422, 550.19, 0.39739],
                [478.71, 493.64, 552.353, 716.5699999999999, 0.33771]
            ],
            "kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 12, "get_state": [[1455.7167590735019, 389.9308521095137, 1593.3395129548937, 804.8326703780252]], "time_since_update": 1, "id": 0, "kf": {"x": [[1524.5281360141978], [597.3817612437695], [57099.930820509384], [0.33169956799834216], [9.641498354998095], [0.2512390527014783], [1261.1879477256625]]}}, {"time_since_observed": 0, "confidence": 0.5, "age": 12, "get_state": [[586.950585237224, 446.0025872289186, 674.1783178789447, 709.6749115280907]], "time_since_update": 1, "id": 1, "kf": {"x": [[630.5644515580843], [577.8387493785046], [22999.53900898925], [0.33081868896770894], [0.0892406806373982], [0.5203642914075408], [-303.28790311845853]]}}, {"time_since_observed": 0, "confidence": 0.5, "age": 7, "get_state": [[1536.9790160857324, 413.27, 1657.2390160857326, 776.04]], "time_since_update": 1, "id": 2, "kf": {"x": [[1597.1090160857325], [594.655], [43626.720199999996], [0.3315048102103261], [7.289484157179798], [0.0], [0.0]]}}],
            "unmatched_trks": [1],
            "occluded_trks": [2],
            "matched": [[0, 0], [1, 1], [2, 2]],
            "trks": [[1455.7167590735019, 389.9308521095137, 1593.3395129548937, 804.8326703780252, 0.0], [586.950585237224, 446.0025872289186, 674.1783178789447, 709.6749115280907, 0.0], [1536.9790160857324, 413.27, 1657.2390160857326, 776.04, 0.0]],
            "ret_kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 12, "get_state": [[1455.7167590735019, 389.9308521095137, 1593.3395129548937, 804.8326703780252]], "time_since_update": 1, "id": 0, "kf": {"x": [[1524.5281360141978], [597.3817612437695], [57099.930820509384], [0.33169956799834216], [9.641498354998095], [0.2512390527014783], [1261.1879477256625]]}}, {"time_since_observed": 0, "confidence": 0.5, "age": 12, "get_state": [[586.950585237224, 446.0025872289186, 674.1783178789447, 709.6749115280907]], "time_since_update": 1, "id": 1, "kf": {"x": [[630.5644515580843], [577.8387493785046], [22999.53900898925], [0.33081868896770894], [0.0892406806373982], [0.5203642914075408], [-303.28790311845853]]}}, {"time_since_observed": 0, "confidence": 0.5, "age": 7, "get_state": [[1536.9790160857324, 413.27, 1657.2390160857326, 776.04]], "time_since_update": 1, "id": 2, "kf": {"x": [[1597.1090160857325], [594.655], [43626.720199999996], [0.3315048102103261], [7.289484157179798], [0.0], [0.0]]}}],
            "ret_unmatched_trks_pos": [[586.950585237224, 446.0025872289186, 674.1783178789447, 709.6749115280907, 2.0]],
            "ret_trks": [
                [1455.7167590735019, 389.9308521095137, 1593.3395129548937, 804.8326703780252, 0.0],
                [586.950585237224, 446.0025872289186, 674.1783178789447, 709.6749115280907, 0.0],
                [1536.9790160857324, 413.27, 1657.2390160857326, 776.04, 0.0]
            ]})

    def test__build_new_targets(self):
        def testAndExpect(x):
            kalman_trackers = list(map(lambda t: DictObj(t), x['kalman_trackers']))
            _build_new_targets(
                x['unmatched_before_before'],
                x['unmatched_before'],
                x['unmatched'],
                x['area_avg'],
                kalman_trackers
            )
            self.assertListEqual(x['unmatched_before_before'], x['ret_unmatched_before_before'])
            self.assertListEqual(x['unmatched_before'], x['ret_unmatched_before'])
            self.assertListEqual(x['unmatched'], x['ret_unmatched'])
            self.assertEqual(x['area_avg'], x['ret_area_avg'])

            self.assertListEqual(
                list(map(lambda t: DictObj(kalman_box_tracker_to_json_no_state(t)) if isinstance(t,
                                                                                                 KalmanBoxTracker) else t,
                         kalman_trackers)),
                list(map(lambda t: DictObj(t), x['ret_kalman_trackers'])))

        testAndExpect({"unmatched_before_before": [[1.0, 389.14, 149.29, 836.0, 0.6479],
                                                   [1198.9, 446.72, 1232.7220000000002, 550.19, 0.5088]],
                       "unmatched_before": [[30.857, 389.14, 179.147, 836.0, 0.78032]],
                       "unmatched": [[30.857, 389.14, 179.147, 836.0, 1.2889]], "area_avg": 81108.71979844959,
                       "kalman_trackers": [{"time_since_observed": 7, "confidence": 1, "age": 314},
                                           {"time_since_observed": 0, "confidence": 0.3447157409799234, "age": 232},
                                           {"time_since_observed": 0, "confidence": 1, "age": 230},
                                           {"time_since_observed": 8, "confidence": 1, "age": 159},
                                           {"time_since_observed": 0, "confidence": 0.5, "age": 121},
                                           {"time_since_observed": 1, "confidence": 0.6744409583467609, "age": 106},
                                           {"time_since_observed": 0, "confidence": 0.5, "age": 95}],
                       "ret_unmatched_before_before": [[1198.9, 446.72, 1232.7220000000002, 550.19, 0.5088]],
                       "ret_unmatched_before": [], "ret_unmatched": [], "ret_area_avg": 81108.71979844959,
                       "ret_kalman_trackers": [{"time_since_observed": 7, "confidence": 1, "age": 314},
                                               {"time_since_observed": 0, "confidence": 0.3447157409799234, "age": 232},
                                               {"time_since_observed": 0, "confidence": 1, "age": 230},
                                               {"time_since_observed": 8, "confidence": 1, "age": 159},
                                               {"time_since_observed": 0, "confidence": 0.5, "age": 121},
                                               {"time_since_observed": 1, "confidence": 0.6744409583467609, "age": 106},
                                               {"time_since_observed": 0, "confidence": 0.5, "age": 95},
                                               {"time_since_observed": 0, "confidence": 0.5, "age": 0}]});
        testAndExpect({"unmatched_before_before": [[1254.6, 446.72, 1288.422, 550.19, 0.5207],
                                                   [729.81, 446.86, 771.6809999999999, 574.47, 0.49008]],
                       "unmatched_before": [[1254.6, 446.72, 1288.422, 550.19, 0.9907],
                                            [729.81, 446.86, 771.6809999999999, 574.47, 0.52778],
                                            [481.15, 464.01, 565.8919999999999, 720.24, 0.45687]],
                       "unmatched": [[1254.6, 446.72, 1288.422, 550.19, 0.89436],
                                     [478.86, 460.48, 569.756, 735.1700000000001, 0.48132],
                                     [729.81, 446.86, 771.6809999999999, 574.47, 0.33441]],
                       "area_avg": 46961.41488824861,
                       "kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 15},
                                           {"time_since_observed": 0, "confidence": 0.5, "age": 15},
                                           {"time_since_observed": 0, "confidence": 0.5, "age": 10}],
                       "ret_unmatched_before_before": [[1254.6, 446.72, 1288.422, 550.19, 0.5207]],
                       "ret_unmatched_before": [[1254.6, 446.72, 1288.422, 550.19, 0.9907],
                                                [481.15, 464.01, 565.8919999999999, 720.24, 0.45687]],
                       "ret_unmatched": [[1254.6, 446.72, 1288.422, 550.19, 0.89436],
                                         [478.86, 460.48, 569.756, 735.1700000000001, 0.48132]],
                       "ret_area_avg": 46961.41488824861,
                       "ret_kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 15},
                                               {"time_since_observed": 0, "confidence": 0.5, "age": 15},
                                               {"time_since_observed": 0, "confidence": 0.5, "age": 10},
                                               {"time_since_observed": 0, "confidence": 0.5, "age": 0}]});

        with open('spec/res/tracker__build_new_targets.json') as f:
            invocations = json.load(f)['invocations']
            for x in invocations:
                testAndExpect(x)

    def test__remove_dead_tracklet(self):
        x = {
                "kalman_trackers": [
                    {"time_since_observed": 0, "confidence": 0.5, "age": 55, "get_state": [[618.7167233320538, 446.82986314705045, 703.5705941709728, 703.3933628232828]], "time_since_update": 0, "id": 1},
                    {"time_since_observed": 6, "confidence": 0.12095521555989777, "age": 40, "get_state": [[1264.2954991567915, 446.5118540805074, 1298.3723086666594, 550.7502364662395]], "time_since_update": 6, "id": 3},
                    {"time_since_observed": 0, "confidence": 0.5, "age": 35, "get_state": [[486.52492777282396, 444.2223193633019, 591.2660093965626, 760.4490279480096]], "time_since_update": 100, "id": 4}
                ],
                "max_age": 3,
                "ret_kalman_trackers": [
                    {"time_since_observed": 0, "confidence": 0.5, "age": 55, "get_state": [[618.7167233320538, 446.82986314705045, 703.5705941709728, 703.3933628232828]], "time_since_update": 0, "id": 1},
                    {"time_since_observed": 6, "confidence": 0.12095521555989777, "age": 40, "get_state": [[1264.2954991567915, 446.5118540805074, 1298.3723086666594, 550.7502364662395]], "time_since_update": 6, "id": 3},
                ],
                "ret": [[[618.7167233320538, 446.82986314705045, 703.5705941709728, 703.3933628232828, 2.0]]]
        }

        kalman_trackers = list(map(lambda t: kalman_tracker_mock_from_json(t), x['kalman_trackers']))
        ret = _remove_dead_tracklet(kalman_trackers, x['max_age'])
        self.assertListEqual(list_of_ndarray_tolist(ret), x['ret'])
        self.assertListEqual(
            list(map(lambda t: kalman_tracker_mock_from_json(t) if isinstance(t, KalmanBoxTracker) else t, kalman_trackers)),
            list(map(lambda t: kalman_tracker_mock_from_json(t), x['ret_kalman_trackers'])))


if __name__ == '__main__':
    unittest.main()
