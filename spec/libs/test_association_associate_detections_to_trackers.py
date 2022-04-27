import json
import unittest

import numpy as np

from libs import association


class DictObj:
    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)

        self.in_dict = in_dict

        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DictObj):
            return False

        for key, value in vars(self).items():
            if o.__getattribute__(key) != value:
                return False
        return True


class TestAssociation(unittest.TestCase):
    def test_empty_detections_and_trackers(self):
        dets = []
        matched, unmatched_dets, unmatched_trks, occluded_trks, unmatched_gts = association.associate_detections_to_trackers(
            None, dets, list(), None, None, iou_threshold=0.3)
        self.assertListEqual(list(), matched.tolist())
        self.assertListEqual(list(), unmatched_dets.tolist())
        self.assertListEqual(list(), unmatched_trks.tolist())
        self.assertListEqual(list(), occluded_trks.tolist())
        self.assertListEqual(list(), unmatched_gts.tolist())

    def test_detections_and_empty_trackers(self):
        dets = [[0, 1, 2, 4], [2, 3, 4, 5]]
        matched, unmatched_dets, unmatched_trks, occluded_trks, unmatched_gts = association.associate_detections_to_trackers(
            None, dets, list(), None, None, iou_threshold=0.3)
        self.assertListEqual(list(), matched.tolist())
        self.assertListEqual([0, 1], unmatched_dets.tolist())
        self.assertListEqual(list(), unmatched_trks.tolist())
        self.assertListEqual(list(), occluded_trks.tolist())
        self.assertListEqual(list(), unmatched_gts.tolist())

    def test_empty_detections_trackers_full(self):
        dets = []
        tracker = [[1, 2, 3, 4]]
        matched, unmatched_dets, unmatched_trks, occluded_trks, unmatched_gts = association.associate_detections_to_trackers(
            None, dets, tracker, None, None, iou_threshold=0.3)
        self.assertListEqual(list(), matched.tolist())
        self.assertListEqual([], unmatched_dets.tolist())
        self.assertListEqual(list(), unmatched_trks.tolist())
        self.assertListEqual(list(), occluded_trks.tolist())
        self.assertListEqual(list(), unmatched_gts.tolist())

    def assert_associate_detections_to_trackers(self, x):
        matched, unmatched_dets, unmatched_trks, occluded_trks, unmatched_gts = association.associate_detections_to_trackers(
            DictObj(x['mot_tracker']), np.array(x['detections']), np.array(x['trackers']), x['groundtruths'],
            x['average_area'], x['iou_threshold'])

        self.assertListEqual(x['ret_matches'], matched.tolist())
        self.assertListEqual(x['ret_unmatched_detections'], unmatched_dets.tolist())
        self.assertListEqual(x['ret_unmatched_trackers'],
                             unmatched_trks if isinstance(unmatched_trks, list) else unmatched_trks.tolist())
        self.assertListEqual(x['ret_occluded_trackers'], occluded_trks.tolist())
        self.assertListEqual(x['ret_unmatched_groundtruths'], unmatched_gts.tolist())

    def test_associate_detections_to_trackers_pass_1(self):
        self.assert_associate_detections_to_trackers({"mot_tracker":{"trackers":[{"time_since_observed":17,"confidence":1,"age":368},{"time_since_observed":14,"confidence":1,"age":273},{"time_since_observed":0,"confidence":1,"age":219},{"time_since_observed":1,"confidence":1,"age":200},{"time_since_observed":0,"confidence":0.07305242074847638,"age":89},{"time_since_observed":0,"confidence":0.5,"age":73},{"time_since_observed":0,"confidence":0.5,"age":72},{"time_since_observed":0,"confidence":0.5,"age":59},{"time_since_observed":0,"confidence":0.2658236378313566,"age":45}],"frame_count":588,"min_hits":3,"conf_trgt":0.35,"conf_objt":0.75,"unmatched_before":[[549.75,429.71,591.621,557.3199999999999,0.42543]]},"detections":[[721.23,438.28,763.101,565.89,1.8465],[585.82,449.36,617.3100000000001,545.83,1.5789],[1085.4,423.72,1176.296,698.4100000000001,1.5448],[1232.4,442.1,1323.296,716.79,1.261],[625,449,664,568,1.151],[555.37,459.21,582.654,543.063,0.51894],[549.75,429.71,591.621,557.3199999999999,0.32455]],"trackers":[[1763.14973972401,375.60857521554806,2017.9606713207927,1142.0506392053612,0],[1164.1812513483892,420.29061997713836,1249.2239974592244,677.4202811074523,0],[1234.1549899593497,439.2151767991311,1327.5242642314074,721.3242702278906,0],[1017.4534659341475,433.66146589620973,1090.8894501218451,655.9696782352675,0],[585.8126568669982,449.3533147315173,617.3895569059339,546.0907701507795,0],[1089.0090407048692,404.7725134807381,1180.014053198618,679.7837427323002,0],[624.6923113598453,449.67069132486915,663.1892759653186,567.1633865221663,0],[725.626516162457,441.5873402727734,761.8188380744704,552.1721161757416,0],[560.6978829220411,460.27153351692573,586.1698374058055,538.6879691125774,0]],"groundtruths":[[912,484,1009,593,0],[1585,-1,1921,577,0],[1163,441,1196,530,0],[1308,431,1342,549,0],[907,414,937,553,0],[1118,429,1160,574,1],[1062,445,1103,569,1],[996,443,1034,547,1],[989,448,1028,549,1],[900,448,936,549,1],[835,473,887,548,0],[796,476,851,536,0],[382,463,428,576,1],[428,458,457,562,1],[424,457,453,565,1],[417,469,457,569,1],[371,447,413,551,0],[427,457,461,540,1],[1090,414,1195,697,1],[1180,432,1254,677,1],[1236,442,1320,714,1],[992,433,1081,672,1],[630,453,662,557,1],[665,454,702,558,1],[722,448,765,551,1],[668,462,693,534,0],[694,462,713,533,0],[711,479,728,536,0],[587,457,620,543,1],[557,461,584,543,1],[910,408,936,537,0],[545,460,570,534,1],[587,440,606,484,1],[587,447,605,490,1],[565,441,577,471,1],[576,435,589,467,1],[985,458,1009,512,0],[1012,454,1030,515,0],[784,452,811,506,1],[758,481,821,510,0]],"average_area":[33159.49929365327],"iou_threshold":0.3,"ret_matches":[[0,7],[1,4],[2,5],[3,2],[4,6],[5,8]],"ret_unmatched_detections":[6],"ret_unmatched_trackers":[],"ret_occluded_trackers":[0,1,3],"ret_unmatched_groundtruths":[]})

    def test_associate_detections_to_trackers_pass_2(self):
        self.assert_associate_detections_to_trackers({ 'mot_tracker':{ 'trackers':[{ 'time_since_observed':0, 'confidence':0.5, 'age':32 }, { 'time_since_observed':0, 'confidence':0.5, 'age':32 }, { 'time_since_observed':5, 'confidence':0.8034821074946065, 'age':27 }, { 'time_since_observed':0, 'confidence':0.5, 'age':17 }, { 'time_since_observed':0, 'confidence':0.5, 'age':12 }], 'frame_count':33, 'min_hits':3, 'conf_trgt':0.35, 'conf_objt':0.75, 'unmatched_before':[[1450, 429.71, 1491.871, 557.3199999999999, 0.9037]] }, 'detections':[[1613.3, 389.14, 1761.59, 836, 2.5745], [607.51, 442.1, 698.406, 716.79, 1.5007], [1450, 429.71, 1491.871, 557.3199999999999, 1.0982], [497.24, 442.1, 588.136, 716.79, 1.0672], [1254.6, 446.72, 1288.422, 550.19, 0.83953], [672.78, 433.93, 746.423, 656.86, 0.43969]], 'trackers':[[1626.9999442228197, 388.1208918913693, 1776.6864681805498, 839.1892449129183, 0], [586.6684833277264, 387.4658788667176, 699.2957247119476, 727.3942309096743, 0], [1721.75360678811, 390.4019263280605, 1860.0098235582702, 807.1669547031725, 0], [1249.5734942682582, 443.2621949826761, 1284.8884336580843, 551.2218934174055, 0], [498.4135397852741, 445.4433065035752, 591.869205281981, 727.8117020759506, 0]], 'groundtruths':[[912, 484, 1009, 593, 0], [1607, 396, 1799, 842, 1], [602, 443, 690, 714, 1], [1585, -1, 1921, 577, 0], [1163, 441, 1196, 530, 0], [1308, 431, 1342, 549, 0], [907, 414, 937, 553, 0], [1683, 416, 1908, 809, 1], [1055, 483, 1091, 593, 1], [1090, 484, 1122, 598, 1], [718, 491, 763, 573, 0], [679, 492, 731, 597, 0], [731, 461, 759, 535, 0], [1258, 447, 1291, 547, 1], [1012, 441, 1052, 557, 1], [1099, 440, 1137, 548, 1], [929, 435, 972, 549, 1], [474, 444, 588, 736, 1], [655, 461, 724, 657, 1], [1448, 430, 1508, 560, 1], [1555, 436, 1607, 558, 1], [500, 459, 588, 710, 1], [835, 473, 887, 548, 0], [796, 476, 851, 536, 0], [547, 463, 582, 556, 1], [496, 456, 523, 558, 1], [375, 446, 416, 550, 0], [419, 459, 458, 543, 1], [581, 455, 617, 590, 1], [1005, 454, 1037, 531, 1], [640, 456, 679, 587, 1], [698, 462, 725, 529, 0], [712, 477, 731, 534, 0], [733, 503, 763, 554, 0], [910, 408, 936, 537, 0], [710, 516, 749, 582, 0], [679, 528, 725, 607, 0], [985, 459, 1004, 512, 0], [1003, 453, 1021, 514, 0], [578, 427, 598, 470, 1], [595, 424, 613, 466, 1], [1027, 449, 1051, 518, 1], [700, 449, 733, 539, 1]], 'average_area':[38725.18380509759], 'iou_threshold':0.3, 'ret_matches':[[0, 0], [1, 1], [3, 4], [4, 3]], 'ret_unmatched_detections':[5, 2], 'ret_unmatched_trackers':[], 'ret_occluded_trackers':[2], 'ret_unmatched_groundtruths':[] })

    def test_associate_detections_to_trackers_pass_3(self):
        self.assert_associate_detections_to_trackers({"mot_tracker":{"trackers":[{"time_since_observed":2,"confidence":1,"age":47},{"time_since_observed":0,"confidence":0.5,"age":47},{"time_since_observed":0,"confidence":0.15166341194080873,"age":32},{"time_since_observed":0,"confidence":0.5,"age":27},{"time_since_observed":2,"confidence":0.11961138221960334,"age":15}],"frame_count":48,"min_hits":3,"conf_trgt":0.35,"conf_objt":0.75,"unmatched_before":[]},"detections":[[618.34,446.86,703.082,703.09,1.2516],[486.58,444.35,591.14,760.03,0.97536],[579.94,451.29,624.888,588.13,0.31026]],"trackers":[[1786.8317504623678,382.8226477084003,1956.5894503077338,894.1241467639411,0],[617.8662874567835,443.9569042506989,705.9295617350076,710.1546678694592,0],[1255.03806910781,442.17287175903624,1290.9762334980953,551.9952275625799,0],[486.4927226793391,444.3220291921217,591.6950224027397,761.9413603902946,0],[1482.9591357034267,430.0855507067737,1524.6639800075843,557.1891628663964,0]],"groundtruths":[[912,484,1009,593,0],[1743,389,1958,884,1],[621,442,709,718,1],[1585,-1,1921,577,0],[1163,441,1196,530,0],[1308,431,1342,549,0],[907,414,937,553,0],[1054,483,1091,593,1],[1090,484,1122,598,1],[709,486,753,577,0],[679,492,731,597,0],[738,461,772,539,0],[1260,447,1293,547,1],[1008,441,1048,557,1],[1098,440,1136,548,1],[928,435,971,549,1],[480,439,593,756,1],[669,459,742,664,1],[1482,430,1539,559,1],[500,454,588,720,1],[835,473,887,548,0],[796,476,851,536,0],[547,462,582,556,1],[496,456,524,558,1],[374,446,416,550,0],[419,459,458,543,1],[582,455,622,589,1],[1019,455,1051,532,1],[646,456,683,579,1],[697,462,725,529,0],[712,477,731,534,0],[738,500,774,558,0],[910,408,936,537,0],[705,518,744,584,0],[679,528,725,607,0],[610,462,636,536,1],[985,459,1004,512,0],[1003,453,1021,514,0],[578,424,598,467,1],[595,423,613,465,1],[1027,450,1051,519,1],[717,447,750,542,1]],"average_area":[30580.30947756609],"iou_threshold":0.3,"ret_matches":[[0,1],[1,3]],"ret_unmatched_detections":[2],"ret_unmatched_trackers":[4,2],"ret_occluded_trackers":[0],"ret_unmatched_groundtruths":[]})

    def test_from_json_array(self):
        with open('spec/res/associate_detections_to_trackers.json') as f:
            invocations = json.load(f)

            i = 0
            for x in invocations:

                if i == 30:
                    # print(i)
                    pass

                self.assert_associate_detections_to_trackers(x)

                i += 1

    def test__unmatched_detections(self):
        def test_and_expect(t):
            actual = association._unmatched_detections(
                np.array(t['detections']),
                np.array(t['matched_indices'])
            )
            self.assertListEqual(t['ret_unmatched_detections'], actual)

        test_and_expect(
            {"detections": [[267.77, 437.53, 388.03, 800.3, 1.429], [961.31, 412.56, 1131.79, 926.01, 1.3889],
                            [359.91, 423.24, 464.47, 738.9200000000001, 1.1941],
                            [1087.1, 363.04, 1312.37, 1040.8600000000001, 1.0567],
                            [566.69, 408.29, 678.83, 746.7, 0.88242],
                            [936.71, 260.92, 1195.63, 1039.68, 0.38177], [60.714, 359.28, 209.004, 806.14, 0.33426]],
             "matched_indices": [[0, 5], [1, 2], [2, 4], [3, 1], [4, 3], [5, 0]], "ret_unmatched_detections": [6]})
        test_and_expect(
            {"detections": [[267.77, 437.53, 388.03, 800.3, 1.8658], [961.31, 412.56, 1131.79, 926.01, 1.385],
                            [1104.1, 394.97, 1300.08, 984.9200000000001, 1.1512],
                            [566.69, 408.29, 678.83, 746.7, 1.1382],
                            [56.715, 391.01, 195.005, 807.87, 0.79262], [359.91, 444.35, 464.47, 760.03, 0.7547],
                            [988.7, 260.92, 1247.6200000000001, 1039.68, 0.38785],
                            [725.08, 442.23, 780.649, 610.94, 0.31484]],
             "matched_indices": [[0, 5], [1, 2], [2, 1], [3, 3], [5, 4], [6, 0]], "ret_unmatched_detections": [4, 7]})
        test_and_expect({"detections": [[272.53, 430.92, 384.66999999999996, 769.33, 1.5459],
                                        [680.04, 389.02, 800.3, 751.79, 1.1476],
                                        [1041.9, 363.04, 1267.17, 1040.8600000000001, 0.99078],
                                        [570.75, 442.1, 661.646, 716.79, 0.93738],
                                        [872.16, 442.23, 927.7289999999999, 610.94, 0.8402],
                                        [444.35, 444.35, 548.9100000000001, 760.03, 0.39837]],
                         "matched_indices": [[0, 6], [1, 3], [2, 1], [3, 4], [4, 0], [5, 5]],
                         "ret_unmatched_detections": []})
        test_and_expect({"detections": [[272.53, 430.92, 384.66999999999996, 769.33, 1.7644],
                                        [679.82, 408.29, 791.96, 746.7, 1.318],
                                        [570.75, 442.1, 661.646, 716.79, 1.2338],
                                        [1056.6, 381.02, 1266.7199999999998, 1013.38, 1.2235],
                                        [872.16, 442.23, 927.7289999999999, 610.94, 1.0282],
                                        [961.31, 412.56, 1131.79, 926.01, 0.66346],
                                        [434.36, 454.06, 531.852, 748.53, 0.62246],
                                        [936.71, 260.92, 1195.63, 1039.68, 0.4031]],
                         "matched_indices": [[0, 6], [1, 3], [2, 4], [3, 1], [5, 2], [6, 5], [7, 0]],
                         "ret_unmatched_detections": [4]})
        test_and_expect(
            {"detections": [[292.02, 437.53, 412.28, 800.3, 1.9404], [961.31, 412.56, 1131.79, 926.01, 1.7096],
                            [1087.1, 363.04, 1312.37, 1040.8600000000001, 1.1698],
                            [78.976, 416.87, 207.936, 805.75, 1.1452], [566.69, 408.29, 678.83, 746.7, 0.70813],
                            [729.81, 472.58, 771.6809999999999, 600.1899999999999, 0.47918],
                            [386.96, 442.1, 477.856, 716.79, 0.46466], [971.06, 292.02, 1212.57, 1018.56, 0.44398]],
             "matched_indices": [[0, 5], [1, 2], [2, 1], [3, 6], [4, 3], [6, 4], [7, 0]],
             "ret_unmatched_detections": [5]})
        test_and_expect(
            {"detections": [[292.02, 437.53, 412.28, 800.3, 2.0159], [961.31, 412.56, 1131.79, 926.01, 1.633],
                            [78.976, 416.87, 207.936, 805.75, 1.592],
                            [1087.1, 363.04, 1312.37, 1040.8600000000001, 1.2742],
                            [589.31, 408.29, 701.4499999999999, 746.7, 0.96689],
                            [971.06, 292.02, 1212.57, 1018.56, 0.71655],
                            [729.81, 472.58, 771.6809999999999, 600.1899999999999, 0.63224]],
             "matched_indices": [[0, 5], [1, 2], [2, 6], [3, 1], [4, 3], [5, 0], [6, 4]],
             "ret_unmatched_detections": []})
        test_and_expect({"detections": [[317.68, 444.35, 422.24, 760.03, 1.6334],
                                        [544.06, 408.29, 656.1999999999999, 746.7, 1.5398],
                                        [946.52, 394.97, 1142.5, 984.9200000000001, 1.2071],
                                        [1098.8, 381.02, 1308.92, 1013.38, 1.1404],
                                        [988.7, 260.92, 1247.6200000000001, 1039.68, 0.40145]],
                         "matched_indices": [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], "ret_unmatched_detections": []})

    def test__unmatched_trackers(self):
        def test_and_expect(t):
            actual = association._unmatched_trackers(
                np.array(t['trackers']),
                np.array(t['matched_indices'])
            )
            self.assertListEqual(t['ret_unmatched_trackers'], actual)

        test_and_expect(
            {"trackers": [[1790.6476379489334, 373.91066521274126, 1968.5101672757476, 909.523056077836, 0.0],
                          [1505.4633627177443, 391.90041890267406, 1747.0613688975798, 1118.7053699971532, 0.0],
                          [1347.7481219417173, 422.33142659797426, 1558.2940650143782, 1055.9737792924616, 0.0],
                          [1105.1864132273108, 423.09464145899426, 1163.7721975747336, 600.8552098264788, 0.0],
                          [932.0860439617203, 439.3534967610465, 1025.287878045643, 720.958009856063, 0.0],
                          [721.7384330895677, 423.8494779654868, 812.5282482908349, 698.2185814162792, 0.0],
                          [1351.5896356396593, 343.87800359776406, 1629.550948111278, 1179.7638198412108, 0.0],
                          [572.7173986943344, 452.95222766192836, 600.7062398777271, 538.9719657939457, 0.0]],
             "matched_indices": [[0, 2], [1, 0], [2, 5], [3, 1], [4, 3], [5, 4], [6, 6]],
             "ret_unmatched_trackers": [7]})
        test_and_expect(
            {"trackers": [[1799.3023095674553, 373.7659292754023, 1977.167985492049, 909.3877957549861, 0.0],
                          [1511.1885309670831, 392.01828807266287, 1752.7865371481664, 1118.8232391708957, 0.0],
                          [1349.1707525520892, 428.2752206607351, 1550.9920214497497, 1035.7522250216414, 0.0],
                          [1113.2440037497245, 420.2284774373579, 1171.7052187433192, 597.6110783613019, 0.0],
                          [942.0050448808294, 441.3301280596905, 1033.4717423529773, 717.7302016822816, 0.0],
                          [721.3794964644474, 423.8506853167762, 812.1704625821583, 698.2232689763503, 0.0],
                          [1300.6714301980162, 339.8734833488548, 1578.3895405838223, 1175.0211782544206, 0.0],
                          [572.819969641354, 452.79952595261244, 600.9965741918049, 539.3963281924049, 0.0],
                          [546.6474850952587, 456.6081768371947, 573.6695149047413, 539.8028231628051, 0.0]],
             "matched_indices": [[0, 2], [1, 5], [2, 1], [3, 0], [4, 8], [5, 6], [6, 7], [7, 4], [8, 3]],
             "ret_unmatched_trackers": []})
        test_and_expect(
            {"trackers": [[1807.957767845859, 373.6235622731163, 1985.8250170484685, 909.2501664970832, 0.0],
                          [1516.9136992167341, 392.13615724359005, 1758.5117053984409, 1118.9411083436999, 0.0],
                          [1358.1033640428993, 424.6400589736668, 1565.3794317670943, 1048.4773535850961, 0.0],
                          [1108.7269681917162, 425.86234198927093, 1164.94231545553, 596.5059129154392, 0.0],
                          [944.4697182193237, 441.82942869881333, 1035.509938970747, 716.9499359186024, 0.0],
                          [735.4283376555388, 427.2884090872653, 822.39063738044, 690.1778828813929, 0.0],
                          [1303.9794013338467, 342.65700250926017, 1581.720616842016, 1177.8741786224264, 0.0],
                          [572.4396109658231, 453.3202882687032, 599.9737948929811, 537.9422781373245, 0.0],
                          [541.9631566574607, 465.4207993593844, 564.8879202656163, 536.090892948308, 0.0]],
             "matched_indices": [[0, 2], [1, 5], [2, 1], [3, 0], [4, 3], [5, 7], [6, 4]],
             "ret_unmatched_trackers": [6, 8]})

    def test__filter_low_iou(self):
        def test_and_expect(t):
            kalman_tracker_obj = list(map(lambda x: DictObj(x), t['kalman_trackers']))
            actual = association._filter_low_iou(
                np.array(t['matched_indices']),
                np.array(t['iou_matrix']),
                t['iou_threshold'],
                np.array(t['unmatched_detections']),
                np.array(t['unmatched_trackers']),
                kalman_tracker_obj
            )
            self.assertListEqual(list(map(lambda x: DictObj(x), t['ret_kalman_trackers'])), kalman_tracker_obj)
            self.assertListEqual(t['ret_matches'], list(map(lambda x: x.tolist(), actual)))

        test_and_expect({"matched_indices": [[0, 0], [1, 1], [3, 2]],
                         "iou_matrix": [[0.8493698835372925, 0.0, 0.20014996826648712], [0.0, 0.9827505350112915, 0.0],
                                        [0.0, 0.0, 0.0], [0.30149003863334656, 0.0, 0.843207061290741], [0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0]], "iou_threshold": 0.3, "unmatched_detections": [2, 4, 5],
                         "unmatched_trackers": [],
                         "kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 14},
                                             {"time_since_observed": 0, "confidence": 0.5, "age": 14},
                                             {"time_since_observed": 0, "confidence": 0.5, "age": 9}],
                         "ret_kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 14},
                                                 {"time_since_observed": 0, "confidence": 0.5, "age": 14},
                                                 {"time_since_observed": 0, "confidence": 0.5, "age": 9}],
                         "ret_matches": [[[0, 0]], [[1, 1]], [[3, 2]]]})
        test_and_expect({"matched_indices": [[0, 0], [1, 1], [4, 2]],
                         "iou_matrix": [[0.8876456618309021, 0.0, 0.21911640465259552], [0.0, 0.962422251701355, 0.0],
                                        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                                        [0.2594248950481415, 0.0, 0.8539206385612488]], "iou_threshold": 0.3,
                         "unmatched_detections": [2, 3], "unmatched_trackers": [],
                         "kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 13},
                                             {"time_since_observed": 0, "confidence": 0.5, "age": 13},
                                             {"time_since_observed": 0, "confidence": 0.5, "age": 8}],
                         "ret_kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 13},
                                                 {"time_since_observed": 0, "confidence": 0.5, "age": 13},
                                                 {"time_since_observed": 0, "confidence": 0.5, "age": 8}],
                         "ret_matches": [[[0, 0]], [[1, 1]], [[4, 2]]]})
        test_and_expect({"matched_indices": [[0, 0], [1, 1]],
                         "iou_matrix": [[0.6649591326713562, 0.0], [0.0, 0.5956258177757263], [0.0, 0.0]],
                         "iou_threshold": 0.3, "unmatched_detections": [2], "unmatched_trackers": [],
                         "kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 3},
                                             {"time_since_observed": 0, "confidence": 0.5, "age": 3}],
                         "ret_kalman_trackers": [{"time_since_observed": 0, "confidence": 0.5, "age": 3},
                                                 {"time_since_observed": 0, "confidence": 0.5, "age": 3}],
                         "ret_matches": [[[0, 0]], [[1, 1]]]})

    def test__clean_up_to_delete(self):
        with open('spec/res/associate__clean_up_to_delete.json') as f:
            invocations = json.load(f)

            i = 0
            for x in invocations:
                matches, unmatched_trackers, unmatched_detections = association._clean_up_to_delete(
                    x['matches_ext_com'], np.array(x['unmatched_detections']), np.array(x['unmatched_trackers']),
                    x['matches'], x['unmatched_before'])

                self.assertListEqual(x['ret_matches'], matches.tolist())
                self.assertListEqual(x['ret_unmatched_detections'], unmatched_detections.tolist())
                self.assertListEqual(x['ret_unmatched_trackers'],
                                     unmatched_trackers if isinstance(unmatched_trackers,
                                                                      list) else unmatched_trackers.tolist())

                i += 1

    def test__occluded_trackers(self):
        with open('spec/res/association__occluded_tracker.json') as f:
            invocations = json.load(f)

            i = 0
            for x in invocations:
                kalman_tracker_obj = list(map(lambda t: DictObj(t), x['kalman_trackers']))
                occluded_trackers, unmatched_trackers = association._occluded_trackers(
                    x['frame_count'], x['min_hits'], np.array(x['ios_matrix']), np.array(x['unmatched_trackers']),
                    np.array(x['trackers']), kalman_tracker_obj, x['average_area'], x['conf_trgt'], x['conf_objt'])

                self.assertListEqual(x['ret_occluded_trackers'], occluded_trackers)
                self.assertListEqual(x['ret_unmatched_trackers'],
                                     unmatched_trackers if isinstance(unmatched_trackers,
                                                                      list) else unmatched_trackers.tolist())
                expected_lists = list(map(lambda t: DictObj(t), x['ret_kalman_trackers']))
                self.assertEqual(len(kalman_tracker_obj), len(expected_lists))
                for i in range(len(expected_lists)):
                    self.assertTrue(expected_lists[i], kalman_tracker_obj[i])

                i += 1

    def test__build_iou_matrix_ext(self):
        with open('spec/res/association__build_iou_matrix_ext.json') as f:
            invocations = json.load(f)
            for x in invocations:
                kalman_tracker_obj = list(map(lambda t: DictObj(t), x['kalman_trackers']))
                iou_matrix_ext = association._build_iou_matrix_ext(
                    np.array(x['unmatched_trackers']),
                    np.array(x['unmatched_detections']),
                    np.array(x['detections']),
                    np.array(x['trackers']),
                    kalman_tracker_obj
                )
                self.assertListEqual(x['ret_iou_matrix_ext'], iou_matrix_ext.tolist())

    def test__filter_low_iou_low_area(self):
        with open('spec/res/association__filter_low_iou_low_area.json') as f:
            invocations = json.load(f)
            for x in invocations:
                matches_ext_com, del_ind = association._filter_low_iou_low_area(
                    np.array(x['matched_indices_ext']),
                    np.array(x['matched_indices']),
                    np.array(x['iou_matrix_ext']),
                    x['iou_threshold'],
                    np.array(x['iou_matrix'])
                )

                if len(matches_ext_com) > 0:
                    if isinstance(matches_ext_com[0], list):
                        self.assertListEqual(x['ret_matches_ext_com'], matches_ext_com)
                    else:
                        self.assertListEqual(x['ret_matches_ext_com'], list(map(lambda l: l.tolist(), matches_ext_com)))
                else:
                    self.assertListEqual(x['ret_matches_ext_com'], matches_ext_com)
                self.assertListEqual(x['ret_del_ind'], del_ind.tolist())


if __name__ == '__main__':
    unittest.main()
