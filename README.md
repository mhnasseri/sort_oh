# SORT with Occlusion Handling

## Introduction

This repository contains code for *Simple Online and Realtime Tracking with Occlusion Handling* (SORT_OH).
In the original [SORT](https://github.com/abewley/sort) algorithm, occlusions are not
detected and caused the ID switch and Fragmentation metrics to be increased. In the [Deep SORT](https://github.com/nwojke/deep_sort)
appearance information is integrated based on a deep appearance descriptor to re-identify occluded targets. That leads to
a decrease in the ID Switch metric, but the fragmentation metric is increased because occluded targets were not detected during occlusion. Moreover, using appearance features increased the computation cost and decreased the speed of the algorithm.

In the proposed algorithm, occlusions are detected using only the location and size of bounding boxes and both ID switch and Fragmentation metrics are decreased simultaneously without a considerable increase in computation cost.
See the [arXiv preprint](http://arxiv.org/abs/2103.04147) for more information.

<p>
<img align="center" width="48%" src="https://github.com/mhnasseri/sort_oh/blob/main/Video_MOT17-02-SDP.gif">
<img align="center" width="48%" src="https://github.com/mhnasseri/sort_oh/blob/main/Video_MOT17-09-SDP.gif">
</p>

## Dependencies

The code is compatible with Python 3. The following dependencies are
needed to run the tracker:

* NumPy
* sklearn
* OpenCV
* filterpy

## Installation

First, clone the repository:
```
git clone https://github.com/mhnasseri/sort_oh.git
```
Then, download the MOT17 dataset from [here](https://motchallenge.net/data/MOT17/).

In order to results on MOT16 be comparable with the results of SORT and DeepSORT algorithms, the private detection taken from the following paper:
```
F. Yu, W. Li, Q. Li, Y. Liu, X. Shi, J. Yan. POI: Multiple Object Tracking with
High Performance Detection and Appearance Feature. In BMTT, SenseTime Group
Limited, 2016.
```
These detections can be downloaded from [here](https://drive.google.com/file/d/0B5ACiy41McAHMjczS2p0dFg3emM/view).

## Running the tracker

To run the code, first select the name of sequences at the beginning of __main__. In the code, the MOT17 sequences for the training phase are selected and the MOT17 sequences for the test phase are commented on. The `phase` variable should be according to the selected sequences. By default the `phase = 'train'`. For running test sequences it should be changed to `phase = 'test'`.

The address of where the dataset is placed should be put in the `mot_path` variable. Also, the path to the outputs of the algorithm should be placed in the `outFolder` variable. Besides, the address to the image and video outputs of the algorithm should be written in the `imgFolder` variable. All these variables are defined after the `phase` variable.

Also, two important parameters in the occlusion handling step of the algorithm, that are the confidence threshold for occlusion by other targets, `conf_trgt`, and the confidence threshold for occlusion by other objects, `conf_objt`, can be set after `phase` variable.

## Evaluating the Results
 To evaluate the results, the development kit that is provided by the motchallenge can be used. It has two different implementations. One is implemented by python and can be accessed [here](https://motchallenge.net/devkit/) The other is implemented using MATLAB and can be accessed [here](https://bitbucket.org/amilan/motchallenge-devkit/)

 The output files of the algorithm should be passed to this development kit alongside the ground truth. This development kit calculates different metrics such as MOTA, MOTP, FP, FN, IDS, FM, ML, MT.

## Main Results

The results of the proposed algorithm on the test dataset of MOT16, alongside the results of SORT and DeepSORT algorithms, are shown in the following table. The private detections from POI paper are used.

Algorithm | MOTA | MOTP | MT | ML | IDS | FM | FP | FN | FPS
----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:
SORT | 59.8 | 79.6 | 25.4% | 22.7% | 1423 | 1835 | 8698 | 63245 | 60
DeepSORT | 61.4 | 79.1 | 32.8% | 18.2% | 781 | 2008 | 12852 | 56668 | 40
SORT_OH | 61.1 | 79.04 | 31.62% | 21.34% | 848 | 1331 | 12296 | 57738 | 162.7

The results of proposed the algorithm on the test dataset of MOT17, are separated for different detection sets. For Comparison, the results of two state-of-the-art algorithms, [Tractor](https://arxiv.org/abs/1903.05625), [GCNNMatch](https://arxiv.org/abs/2010.00067), are included in the following table.

Algorithm | Detection| MOTA | MT | ML | IDS | FM | FP | FN | FPS
----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:
Tractor | ALL | 53.5 | 19.5% | 36.6% | 2072 | 4611 | 12201 | 248047 | 1.5
GCNNMatch | ALL | 57.0 | 23.3% | 34.6% | 1957 | 2798 | 12283 | 228242 | 1.3
SORT_OH | ALL | 44.3 | 14.4% | 45.6% | 2191 | 5243 | 21796 | 290065 | 137.5
Tractor | DPM | 52.2 | 14.9% | 37.5% | 635 | - | 2908 | 86275 | -
GCNNMatch | DPM | 55.5 | 21.5% | 37.6% | 564 | 782 | 2937 | 80242 | -
SORT_OH | DPM | 29.7 | 5.5% | 64.7% | 530 | 1448 | 3048 | 128696 | -
Tractor | FRCNN | 52.9 | 16.2% | 34.7% | 648 | - | 3918 | 83904 | -
GCNNMatch | FRCNN | 56.1 | 22.3% | 33.9% | 647 | 934 | 4015 | 77950 | -
SORT_OH | FRCNN | 44.9 | 14.7% | 40.3% | 795 | 1571 | 9102 | 93669 | -
Tractor | SDP | 55.3 | 18.1% | 32.9% | 789 | - | 5375 | 77868 | -
GCNNMatch | SDP | 59.5 | 26.0% | 32.4% | 746 | 1082 | 5331 | 70050 | -
SORT_OH | SDP | 58.4 | 22.8% | 32.0% | 866 | 2224 | 9646 | 67700 | -

## Highlevel overview of source files

In the top-level directory are executable scripts to execute, and visualize the tracker. The main entry point is in `tracker_app.py`.
This file runs the tracker on a MOTChallenge sequence.

In folder `libs` are written libraries that are used in tracking code:

* `association.py`: Associate detections to the targets and detects new targets. Association is done in a cascade manner. At first, detections are matched to targets using IoU measure. Then the bounding box of occluded unmatched targets are extended and they are matched with unmatched detections of the previous step using extended IoU measure. In the end, unmatched targets with high confidence measures are marked as occluded. (associate_detections_to_trackers() function) Detecting new targets is done using unmatched detections of current, previous, and two previous frames. (find_new_trackers_2() function)   
* `calculate_cost.py`: Calculate association measures. There are functions for calculating different kinds of Intersection over Union(IoU) between two bounding boxes. (iou(), iou_ext(), iou_ext_sep(), ios() functions) Using these functions, IoU matrix is calculated between two sets of bounding boxes. (cal_iou(), cal_ios() functions)
* `convert.py`: convert the format of a bounding box from [x1, y1, x2, y2] to [x, y, s, r] and vice versa.
* `kalman_tracker.py`: Initialize a Kalman Filter for every new target. Then predict its location and correct its estimation A Kalman filter with a constant velocity model is initialized for every new target. (KalmanBoxTracker()) This filter is used for predicting the location of the target in the new frame. (predict()) The predicted location is corrected if it is matched with a detection. (update())
* `tracker.py`: The high-level implementation of SORT_OH algorithm. For every sequence, a SORT_OH tracker is initialized. It keeps track of different parameters of the tracker, such as targets, unmatched detections of current, previous, and two previous frames. At first, the location of all targets in the previous frame is predicted using the specific Kalman filter of every target. Then targets are associated with the detections. In the end, new targets are detected, and exited targets are removed.
* `visuallization.py`: Used for drawing bounding boxes of targets and detections in every frame and generating video for every sequence. By changing DisplayState class variables, drawing different bounding boxes is enabled or disabled. The detections are drawn with a thin black rectangle. The targets are drawn with colored thick rectangles. The extended bounding boxes are shown with dashed colored rectangles and ground truths are shown with thin red rectangles.   

The `tracker_app.py` expects detections in motchallenge format

## Citing SORT_OH

If you find this repo useful in your research, please consider citing the following paper:

    @article{nasseri2021simple,
      title={Simple online and real-time tracking with occlusion handling},
      author={Nasseri, Mohammad Hossein and Moradi, Hadi and Hosseini, Reshad and Babaee, Mohammadreza},
      journal={arXiv preprint arXiv:2103.04147},
      year={2021}
    }
