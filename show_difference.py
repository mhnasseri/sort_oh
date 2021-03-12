import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from libs import association
from PIL import Image


if __name__ == '__main__':
    # train sequences
    # sequences = ['MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', 'MOT17-09-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM',
    #              'MOT17-13-DPM']
    # sequences = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN',
    #              'MOT17-11-FRCNN', 'MOT17-13-FRCNN']
    sequences = ['MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP',
                 'MOT17-11-SDP', 'MOT17-13-SDP']
    # test sequences
    # sequences = ['MOT17-01-DPM', 'MOT17-03-DPM', 'MOT17-06-DPM', 'MOT17-07-DPM', 'MOT17-08-DPM',
    #              'MOT17-12-DPM', 'MOT17-14-DPM']
    # sequences = ['MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN', 'MOT17-07-FRCNN', 'MOT17-08-FRCNN',
    #              'MOT17-12-FRCNN', 'MOT17-14-FRCNN']
    # sequences = ['MOT17-01-SDP', 'MOT17-03-SDP', 'MOT17-06-SDP', 'MOT17-07-SDP', 'MOT17-08-SDP',
    #               'MOT17-12-SDP', 'MOT17-14-SDP']
    phase = 'train'
    imgFolder = 'images/img_conf_vs_before'
    outFolder1 = 'outputs/output_OH_EX_0.2_N_SEP_ShowEx_1'
    outFolder2 = 'outputs/output_OH_EX_0.2_N_SEP_CONF_0.15_0.5'
    mot_path = 'C:/Users/mhnas/Courses/Thesis/MOT/MOT17'

    if not os.path.exists(outFolder1) or not os.path.exists(outFolder2):
        print('check the paths for output files')
    else:
        for seq in sequences:
            common_path = ('%s/%s/%s' % (mot_path, phase, seq))
            seq_trk_1 = np.loadtxt('%s/%s.txt' % (outFolder1, seq), delimiter=',')  # load trackers of file 1
            seq_trk_2 = np.loadtxt('%s/%s.txt' % (outFolder2, seq), delimiter=',')  # load trackers of file 2

            if not os.path.exists('%s/%s' % (common_path, imgFolder)):
                os.makedirs('%s/%s' % (common_path, imgFolder))

            for frame in tqdm(range(int(seq_trk_1[:, 0].max()))):
                frame += 1  # detection and frame numbers begin at 1
                trks1 = seq_trk_1[seq_trk_1[:, 0] == frame, 2:7]
                trks2 = seq_trk_2[seq_trk_2[:, 0] == frame, 2:7]
                trks1[:, 2:4] += trks1[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                trks2[:, 2:4] += trks2[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                matches = association.find_new_trackers(trks1, trks2)
                unmatched_trks1 = []
                for t1, trk1 in enumerate(trks1):
                    if t1 not in matches[:, 0]:
                        unmatched_trks1.append(trk1)
                unmatched_trks2 = []
                for t2, trk2 in enumerate(trks2):
                    if t2 not in matches[:, 0]:
                        unmatched_trks2.append(trk2)

                img = Image.open('%s/img1/%06d.jpg' % (common_path, frame))
                im = np.array(img, dtype=np.uint8)
                fig, ax = plt.subplots(1, figsize=(19.5, 11))
                ax.imshow(im)
                for m in range(len(matches)):
                    ind1 = matches[m, 0]
                    ind2 = matches[m, 1]
                    trk_avg = (trks1[ind1, 0:4] + trks2[ind2, 0:4])/2
                    rect = patches.Rectangle((trk_avg[0], trk_avg[1]), trk_avg[2] - trk_avg[0], trk_avg[3] - trk_avg[1],
                                             linewidth=2, edgecolor='green', facecolor='none')
                    ax.add_patch(rect)

                for ut1 in unmatched_trks1:
                    rect = patches.Rectangle((ut1[0], ut1[1]), ut1[2] - ut1[0], ut1[3] - ut1[1], linewidth=3,
                                             edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                for ut2 in unmatched_trks2:
                    rect = patches.Rectangle((ut2[0], ut2[1]), ut2[2] - ut2[0], ut2[3] - ut2[1], linewidth=3,
                                             edgecolor='blue', facecolor='none')
                    ax.add_patch(rect)

                fig.tight_layout()
                fig.savefig('%s/%s/%06d.jpg' % (common_path, imgFolder, frame))
                plt.close(fig)

    print('Showing difference is finished')


