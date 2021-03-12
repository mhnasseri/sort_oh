from __future__ import print_function
import os.path
import numpy as np
import time
import argparse
from tqdm import tqdm

# new imports
from libs import visualization
from libs import tracker
from libs import kalman_tracker


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT with occlusion handling demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # train sequences
    sequences = ['MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', 'MOT17-09-DPM', 'MOT17-10-DPM',
                 'MOT17-11-DPM', 'MOT17-13-DPM',
                 'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN',
                 'MOT17-11-FRCNN', 'MOT17-13-FRCNN',
                 'MOT17-09-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP',
                 'MOT17-11-SDP', 'MOT17-13-SDP']
    # sequences = ['MOT17-02-POI', 'MOT17-04-POI', 'MOT17-05-POI', 'MOT17-09-POI', 'MOT17-10-POI',
    #              'MOT17-11-POI', 'MOT17-13-POI']
    # # test sequences
    # sequences = ['MOT17-01-DPM', 'MOT17-03-DPM', 'MOT17-06-DPM', 'MOT17-07-DPM', 'MOT17-08-DPM',
    #              'MOT17-12-DPM', 'MOT17-14-DPM',
    #              'MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN', 'MOT17-07-FRCNN', 'MOT17-08-FRCNN',
    #              'MOT17-12-FRCNN', 'MOT17-14-FRCNN',
    #              'MOT17-01-SDP', 'MOT17-03-SDP', 'MOT17-06-SDP', 'MOT17-07-SDP', 'MOT17-08-SDP',
    #              'MOT17-12-SDP', 'MOT17-14-SDP']
    # sequences = ['MOT17-01-POI', 'MOT17-03-POI', 'MOT17-06-POI', 'MOT17-07-POI', 'MOT17-08-POI',
    #              'MOT17-12-POI', 'MOT17-14-POI']
    phase = 'train'     # 'test'
    # for conf_t in range(50, 90, 5):
    #     for conf_o in range(conf_t, 100, 5):
    #         conf_trgt = conf_t/100
    #         conf_objt = conf_o/100
    #         print('conf_target = ', conf_trgt, ', conf_objects = ', conf_objt)
    conf_trgt = 0.35
    conf_objt = 0.75
    imgFolder = 'images/img_OH_EX_0.2_N_SEP_CONF_TOT_MIN_EXT_%s_%s' % (conf_trgt, conf_objt)     # Occlusion Handling + Area Cost /Extended Separate
    outFolder = 'outputs/output_OH_EX_0.2_N_SEP_CONF_TOT_MIN_%s_%s' % (conf_trgt, conf_objt)
    mot_path = 'C:/Users/mhnas/Courses/Thesis/MOT/MOT17'
    args = parse_args()

    total_time = 0.0
    total_frames = 0

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    for seq in sequences:
        kalman_tracker.KalmanBoxTracker.count = 0   # Make zero ID number in the new sequence
        mot_tracker = tracker.Sort_OH()  # create instance of the SORT with occlusion handling tracker
        mot_tracker.seq = seq
        mot_tracker.conf_trgt = conf_trgt
        mot_tracker.conf_objt = conf_objt
        common_path = ('%s/%s/%s' % (mot_path, phase, seq))
        seq_dets = np.loadtxt('%s/det/det.txt' % common_path, delimiter=',')  # load detections
        if phase == 'train':
            seq_gts = np.loadtxt('%s/gt/gt.txt' % common_path, delimiter=',')  # load ground truth

        with open('%s/%s.txt' % (outFolder, seq), 'w') as out_file:
            print("\nProcessing %s." % seq)

            if not os.path.exists('%s/%s' % (common_path, imgFolder)):
                os.makedirs('%s/%s' % (common_path, imgFolder))

            for frame in tqdm(range(int(seq_dets[:, 0].max()))):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]

                # remove dets with low confidence
                for d in reversed(range(len(dets))):
                    if dets[d, 4] < 0.3:
                        dets = np.delete(dets, d, 0)

                dets[:, 2:4] += dets[:, 0:2]  # convert [x1,y1,w,h] to [x1,y1,x2,y2]
                gts = []
                if phase == 'train':
                    gts = seq_gts[seq_gts[:, 0] == frame, 2:7]
                    gts[:, 2:4] += gts[:, 0:2]  # convert [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                start_time = time.time()
                trackers, unmatched_trckr, unmatched_gts = mot_tracker.update(dets, gts)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                # save trackers -> Frame number, ID, Left, Top, Width, Height
                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)

                visualization.dispaly_details(common_path, imgFolder, frame, dets, gts, trackers,
                                              unmatched_trckr, unmatched_gts, mot_tracker.trackers)
        # visualization.generate_video('%s/%s' % (common_path, imgFolder), '%s/%s/Video_%s.avi' % (common_path, imgFolder, seq), 10)

    # save run result in a file
    with open('%s/summery_%s.txt' % (outFolder, phase), 'w') as sum_file:
        print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time),
              file=sum_file)

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
    if visualization.DisplayState.display:
        print("Note: to get real runtime results run without the option: --display")

