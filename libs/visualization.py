import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os.path
import cv2
from PIL import Image
from . import convert


colours_rand = np.random.rand(32, 3)  # used only for display
# antiquewhite, beige, bisque, blanchedalmond, azure, black, aliceblue
colours = [
    'aqua', 'aquamarine', 'blue', 'crimson', 'brown', 'burlywood', 'yellow',
    'orange', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'blueviolet', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'chartreuse', 'red', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellowgreen'
]
clrs = ['r', 'b', 'g', 'm', 'c', 'y', 'w', 'k']  # used for displaying center of bounding boxes


class DisplayState:
    display = True
    display_trgts = True
    display_unmatched_trks = True
    display_extended = True
    display_dets = True
    display_gt = False
    display_gt_diff = False
    display_trgt_id = False
    display_det_uncer = False


# display details such as bounding boxes in the image
def dispaly_details(common_path, imgFolder, frame, dets, gts, trackers,
                    unmatched_trckr, unmatched_gts, mot_trackers):
    # display image
    if DisplayState.display:
        img = Image.open('%s/img1/%06d.jpg' % (common_path, frame))
        im = np.array(img, dtype=np.uint8)
        fig, ax = plt.subplots(1, figsize=(19.5, 11))
        ax.imshow(im)

        # display trackers
        if DisplayState.display_trgts:
            for t in trackers:
                rect = patches.Rectangle((t[0], t[1]), t[2] - t[0], t[3] - t[1], linewidth=5,
                                         edgecolor=colours[int(t[4]) % 50], facecolor='none')
                ax.add_patch(rect)
                # display target id
                if DisplayState.display_trgt_id:
                    plt.text(t[0], t[1] - 5, int(t[4]), fontsize=14, bbox={'facecolor': colours[int(t[4]) % 50], 'alpha': 0.5, 'pad': 2})

        # display unmatched trackers
        if DisplayState.display_unmatched_trks:
            for ut in unmatched_trckr:
                rect = patches.Rectangle((ut[0], ut[1]), ut[2] - ut[0], ut[3] - ut[1], linewidth=2,
                                         edgecolor=colours[int(ut[4]) % 50], linestyle='dashed', facecolor='none')
                ax.add_patch(rect)

        # display extended bounding boxes of occluded targets
        if DisplayState.display_extended:
            for t in mot_trackers:
                if t.time_since_observed > 0:
                    bbx = convert.x_to_bbox(t.kf.x[0:4], None)
                    ex_w = np.minimum(1.2, (t.time_since_observed + 1) * 0.3)
                    ex_h = np.minimum(0.5, (t.time_since_observed + 1) * 0.1)
                    w = (bbx[0, 2] - bbx[0, 0])*(1 + ex_w)
                    h = (bbx[0, 3] - bbx[0, 1])*(1 + ex_h)
                    left_p = bbx[0, 0] - (bbx[0, 2] - bbx[0, 0]) * ex_w / 2
                    top_p = bbx[0, 1] - (bbx[0, 3] - bbx[0, 1]) * ex_h / 2
                    rect = patches.Rectangle((left_p, top_p), w, h, linewidth=2,
                                             edgecolor=colours[int(t.id + 1) % 50], linestyle='dashed', facecolor='none')
                    ax.add_patch(rect)

        # display detections
        if DisplayState.display_dets:
            for d in dets:
                rect = patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], linewidth=2, edgecolor='k',
                                         facecolor='none')
                ax.add_patch(rect)
                # display detection uncertainty
                if DisplayState.display_det_uncer:
                    plt.text(d[0], d[1] + 8, d[4], fontsize=10, bbox={'facecolor': 'k', 'alpha': 0.5, 'pad': 2})

        # display ground truths
        if DisplayState.display_gt:
            for g in gts:
                rect = patches.Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], linewidth=2,
                                         edgecolor='r', linestyle='dashed', facecolor='none')
                ax.add_patch(rect)

        # display unmatched ground truths
        if DisplayState.display_gt_diff:
            for ugt in unmatched_gts:
                rect = patches.Rectangle((ugt[0], ugt[1]), ugt[2] - ugt[0], ugt[3] - ugt[1], linewidth=2,
                                         edgecolor='r', linestyle='dashed', facecolor='none')
                ax.add_patch(rect)

        # save image & enabled bounding boxes
        # to reduce the margins of the image
        fig.tight_layout()
        fig.savefig('%s/%s/%06d.jpg' % (common_path, imgFolder, frame))
        plt.close(fig)


# Video Generating function
def generate_video(image_folder, video_name, fps):
    if DisplayState.display:
        # Array images should only consider the image files ignoring others if any
        images = [img for img in os.listdir(image_folder)
                  if img.endswith(".jpg") or
                  img.endswith(".jpeg") or
                  img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))

        # setting the frame width, height width according to the width & height of first image
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

        # Appending the images to the video one by one
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        # Deallocating memories taken for window creation
        cv2.destroyAllWindows()
        video.release()  # releasing the video generated
        print('Video Generated')
