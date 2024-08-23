#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import json
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc

logger = logging.get_logger(__name__)


def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    Args:
        full_groundtruth(dict): list of groundtruth.
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):
        for key in full_groundtruth[i].keys():
            if int(key.split(",")[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))
        if multi_label:
            self.video_preds -= 1e10  # don't understand

        self.video_labels = (
            torch.zeros((num_videos, num_cls))
            if multi_label
            else torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        if self.multi_label:
            self.video_preds -= 1e10
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5), log_confusion_matrix=False):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}
        if self.multi_label:
            map = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            self.stats["map"] = map
        else:
            num_topks_correct = metrics.topks_correct(
                self.video_preds, self.video_labels, ks
            )
            topks = [
                (x / self.video_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]
            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )

            mean_cls_acc, recall_percls = metrics.mean_class_accuracy(preds=self.video_preds, labels=self.video_labels)
            self.stats["mean_class_acc"] = mean_cls_acc
            if log_confusion_matrix:
                self.stats["confusion_matrix"] = metrics.conf_matrix(preds=self.video_preds, labels=self.video_labels).tolist()
        logging.log_json_stats(self.stats)


class TestGazeMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
            self,
            num_videos,
            num_clips,
            num_cls,
            overall_iters,
            dataset
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options include "sum", and "max".
        """

        self.dataset = dataset
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))

        self.video_labels = torch.zeros((num_videos, num_cls))
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        self.num_samples = 0.0
        self.iou = list()
        self.recall = list()
        self.precision = list()
        self.f1 = list()
        self.auc = list()

        self.preds = list()
        self.labels_hm = list()
        self.labels = list()

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        self.video_labels.zero_()

        self.num_samples = 0.0
        self.iou = list()
        self.recall = list()
        self.precision = list()
        self.f1 = list()
        self.auc = list()

        self.preds = list()
        self.labels_hm = list()
        self.labels = list()

    def update_stats(self, iou, f1, recall, precision, auc, preds, labels_hm, labels):
        labels_flat = labels.view(labels.size(0) * labels.size(1), -1)
        mb_size = torch.where(labels_flat[:, 2] == 1)[0].size(0)
        self.num_samples += mb_size
        self.iou.append(iou * mb_size)
        self.f1.append(f1 * mb_size)
        self.recall.append(recall * mb_size)
        self.precision.append(precision * mb_size)
        self.auc.append(auc * mb_size)

        self.preds.append(preds)
        self.labels_hm.append(labels_hm)
        self.labels.append(labels)

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self):
        # iou = sum(self.iou) / self.num_samples
        # recall = sum(self.recall) / self.num_samples
        # precision = sum(self.precision) / self.num_samples
        # f1 = (2 * recall * precision) / (recall + precision + 1e-6)
        # mean_f1 = sum(self.f1) / self.num_samples

        preds = torch.cat(self.preds, dim=0)
        labels_hm = torch.cat(self.labels_hm, dim=0)
        labels = torch.cat(self.labels, dim=0)
        f1, recall, precision, threshold = metrics.adaptive_f1(preds, labels_hm, labels, dataset=self.dataset)
        iou = metrics.gaze_iou(preds, labels_hm, threshold=threshold)
        aae = metrics.average_angle_error(preds, labels, dataset=self.dataset)
        auc = metrics.auc(preds, labels_hm, labels, dataset=self.dataset)

        # frames_f1 = list()
        # if preds.ndim == 4:  # test of outer model
        #     preds = preds.unsqueeze(1)
        # for i in range(preds.size(2)):
        #     pred = preds[:, :, i:i+1, :, :]
        #     label_hm = labels_hm[:, i:i+1, :, :]
        #     label = labels[:, i:i+1, :]
        #     per_frame_f1, per_frame_recall, per_frame_precision \
        #         = metrics.fixed_f1(pred, label_hm, label, dataset=self.dataset, threshold=threshold)
        #     frames_f1.append(per_frame_f1.item())

        self.stats = {
            "split": "test_final",
            "iou": iou,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "aae": aae,
            "auc": auc,
            "threshold": threshold,
            # "frames_f1": json.dumps(frames_f1)
        }

        logging.log_json_stats(self.stats)


class TestGazeShiftMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))
        if multi_label:
            self.video_preds -= 1e10  # don't understand

        self.video_labels = (torch.zeros((num_videos, num_cls)) if multi_label else torch.zeros((num_videos)).long())
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        if self.multi_label:
            self.video_preds -= 1e10
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(self.video_labels[vid_id].type(torch.FloatTensor), labels[ind].type(torch.FloatTensor),)
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(self.video_preds[vid_id], preds[ind])
            else:
                raise NotImplementedError("Ensemble Method {} is not supported".format(self.ensemble_method))
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, log_confusion_matrix=False):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning("clip count {} ~= num clips {}".format(", ".join(["{}: {}".format(i, k) for i, k in enumerate(self.clip_count.tolist())]), self.num_clips,))

        self.stats = {"split": "test_final"}
        if self.multi_label:
            map = get_map(self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy())
            self.stats["map"] = map
        else:
            f1, recall, precision, threshold = metrics.mean_f1_for_classification(self.video_preds[:, 1], self.video_labels)
            auc = metrics.auc_for_classification(self.video_preds[:, 1], self.video_labels)
            binary_preds = (self.video_preds[:, 1] > threshold).long()
            acc = metrics.accuracy(preds=binary_preds, labels=self.video_labels)
            mean_class_acc, _ = metrics.mean_class_accuracy(preds=torch.stack([1 - binary_preds, binary_preds], dim=1), labels=self.video_labels)
            self.stats["mean f1"] = f1
            self.stats["mean recall"] = recall
            self.stats["mean precision"] = precision
            self.stats["auc"] = auc
            self.stats["acc"] = acc
            self.stats["mean_class_acc"] = mean_class_acc
            self.stats["threshold"] = threshold

            if log_confusion_matrix:
                self.stats["confusion_matrix"] = metrics.conf_matrix(preds=binary_preds, labels=self.video_labels).tolist()
        logging.log_json_stats(self.stats)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        if not self._cfg.DATA.MULTI_LABEL:
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1))
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            avg_loss = self.loss_total / self.num_samples
            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["loss"] = avg_loss
        logging.log_json_stats(stats)


class TrainGazeMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        self.iou = ScalarMeter(cfg.LOG_PERIOD)
        self.iou_total = 0.0
        self.f1 = ScalarMeter(cfg.LOG_PERIOD)
        self.recall = ScalarMeter(cfg.LOG_PERIOD)
        self.recall_total = 0.0
        self.precision = ScalarMeter(cfg.LOG_PERIOD)
        self.precision_total = 0.0
        self.auc = ScalarMeter(cfg.LOG_PERIOD)
        self.auc_total = 0.0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR
        self.threshold = -1

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.iou.reset()
        self.iou_total = 0.0
        self.f1.reset()
        self.recall.reset()
        self.recall_total = 0.0
        self.precision.reset()
        self.precision_total = 0.0
        self.auc.reset()
        self.auc_total = 0.0
        self.num_samples = 0
        self.threshold = -1

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, iou, f1, recall, precision, auc, threshold, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            iou (float): iou metric.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.lr = lr
        self.loss.add_value(loss)
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        # Current minibatch stats
        self.iou.add_value(iou)
        self.iou_total += iou * mb_size
        self.f1.add_value(f1)
        self.recall.add_value(recall)
        self.recall_total += recall * mb_size
        self.precision.add_value(precision)
        self.precision_total += precision * mb_size
        self.auc.add_value(auc)
        self.auc_total += auc * mb_size
        self.threshold = threshold

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1))
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "iou": self.iou.get_win_median(),
            "f1": self.f1.get_win_median(),
            "recall": self.recall.get_win_median(),
            "precision": self.precision.get_win_median(),
            "auc": self.auc.get_win_median(),
            "threshold": self.threshold
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            "loss": self.loss_total / self.num_samples,
            "iou": self.iou_total / self.num_samples,
            "auc": self.auc_total / self.num_samples
        }
        recall = self.recall_total / self.num_samples
        precision = self.precision_total / self.num_samples
        f1 = 2 * recall * precision / (recall + precision + 1e-6)
        stats['f1'] = f1
        stats['recall'] = recall
        stats['precision'] = precision

        logging.log_json_stats(stats)


class TrainGazeShiftMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        self.f1 = ScalarMeter(cfg.LOG_PERIOD)
        self.f1_total = 0.0
        self.recall = ScalarMeter(cfg.LOG_PERIOD)
        self.recall_total = 0.0
        self.precision = ScalarMeter(cfg.LOG_PERIOD)
        self.precision_total = 0.0
        self.auc = ScalarMeter(cfg.LOG_PERIOD)
        self.auc_total = 0.0
        self.acc = ScalarMeter(cfg.LOG_PERIOD)
        self.acc_total = 0.0
        self.mean_cls_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mean_cls_acc_total = 0.0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR
        self.threshold = -1

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.f1.reset()
        self.f1_total = 0.0
        self.recall.reset()
        self.recall_total = 0.0
        self.precision.reset()
        self.precision_total = 0.0
        self.auc.reset()
        self.auc_total = 0.0
        self.acc.reset()
        self.acc_total = 0.0
        self.mean_cls_acc.reset()
        self.mean_cls_acc_total = 0.0
        self.num_samples = 0
        self.threshold = -1

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, f1, recall, precision, auc, acc, mean_cls_acc, threshold, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        # Current minibatch stats
        self.f1.add_value(f1)
        self.recall.add_value(recall)
        self.precision.add_value(precision)
        self.auc.add_value(auc)
        self.acc.add_value(acc)
        self.mean_cls_acc.add_value(mean_cls_acc)

        # Aggregate stats
        self.f1_total += f1 * mb_size
        self.recall_total += recall * mb_size
        self.precision_total += precision * mb_size
        self.auc_total += auc * mb_size
        self.acc_total += acc * mb_size
        self.mean_cls_acc_total += mean_cls_acc * mb_size
        self.threshold = threshold

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1))
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "mean f1": self.f1.get_win_median(),
            "mean recall": self.recall.get_win_median(),
            "mean precision": self.precision.get_win_median(),
            "auc": self.auc.get_win_median(),
            "accuracy": self.acc.get_win_median(),
            "mean_class_acc": self.mean_cls_acc.get_win_median(),
            "threshold": self.threshold
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            "loss": self.loss_total / self.num_samples,
            "mean f1": self.f1_total / self.num_samples,
            "mean recall": self.recall_total / self.num_samples,
            "mean precision": self.precision_total / self.num_samples,
            "auc": self.auc_total / self.num_samples,
            "acc": self.acc_total / self.num_samples,
            "mean_class_acc": self.mean_cls_acc_total / self.num_samples
        }
        logging.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch, log_confusion_matrix=False):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            stats["map"] = get_map(
                torch.cat(self.all_preds).cpu().numpy(),
                torch.cat(self.all_labels).cpu().numpy(),
            )
        else:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            self.min_top1_err = min(self.min_top1_err, top1_err)
            self.min_top5_err = min(self.min_top5_err, top5_err)

            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["min_top1_err"] = self.min_top1_err
            stats["min_top5_err"] = self.min_top5_err

            all_preds = torch.cat(self.all_preds, dim=0)
            all_labels = torch.cat(self.all_labels, dim=0)
            mean_cls_acc, recall_percls = metrics.mean_class_accuracy(preds=all_preds, labels=all_labels)
            stats["mean_class_acc"] = mean_cls_acc
            if log_confusion_matrix:
                stats["confusion_matrix"] = metrics.conf_matrix(preds=all_preds, labels=all_labels).tolist()

        logging.log_json_stats(stats)


class ValGazeMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.iou = ScalarMeter(cfg.LOG_PERIOD)
        self.iou_total = 0.0
        self.f1 = ScalarMeter(cfg.LOG_PERIOD)
        self.recall = ScalarMeter(cfg.LOG_PERIOD)
        self.recall_total = 0.0
        self.precision = ScalarMeter(cfg.LOG_PERIOD)
        self.precision_total = 0.0
        self.auc = ScalarMeter(cfg.LOG_PERIOD)
        self.auc_total = 0.0
        self.threshold = -1
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.iou.reset()
        self.iou_total = 0.0
        self.f1.reset()
        self.recall.reset()
        self.recall_total = 0.0
        self.precision.reset()
        self.precision_total = 0.0
        self.auc.reset()
        self.auc_total = 0.0
        self.threshold = -1
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, iou, f1, recall, precision, auc, labels, threshold):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        labels_flat = labels.view(labels.size(0) * labels.size(1), -1)
        mb_size = torch.where(labels_flat[:, 2] == 1)[0].size(0)
        self.num_samples += mb_size
        self.iou.add_value(iou)
        self.iou_total += iou * mb_size
        self.f1.add_value(f1)
        self.recall.add_value(recall)
        self.recall_total += recall * mb_size
        self.precision.add_value(precision)
        self.precision_total += precision * mb_size
        self.auc.add_value(auc)
        self.auc_total += auc * mb_size
        self.threshold = threshold

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "iou": self.iou.get_win_median(),
            "f1": self.f1.get_win_median(),
            "recall": self.recall.get_win_median(),
            "precision": self.precision.get_win_median(),
            "auc": self.auc.get_win_median(),
            "threshold": self.threshold
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            "iou": self.iou_total / self.num_samples,
            "auc": self.auc_total / self.num_samples
        }
        recall = self.recall_total / self.num_samples
        precision = self.precision_total / self.num_samples
        f1 = 2 * recall * precision / (recall + precision + 1e-6)
        stats['recall'] = recall
        stats['precision'] = precision
        stats['f1'] = f1

        logging.log_json_stats(stats)


class ValGazeShiftMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.f1 = ScalarMeter(cfg.LOG_PERIOD)
        self.f1_total = 0.0
        self.recall = ScalarMeter(cfg.LOG_PERIOD)
        self.recall_total = 0.0
        self.precision = ScalarMeter(cfg.LOG_PERIOD)
        self.precision_total = 0.0
        self.auc = ScalarMeter(cfg.LOG_PERIOD)
        self.auc_total = 0.0
        self.acc = ScalarMeter(cfg.LOG_PERIOD)
        self.acc_total = 0.0
        self.mean_class_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mean_class_acc_total = 0.0
        self.threshold = -1
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.f1.reset()
        self.f1_total = 0.0
        self.recall.reset()
        self.recall_total = 0.0
        self.precision.reset()
        self.precision_total = 0.0
        self.auc.reset()
        self.auc_total = 0.0
        self.acc.reset()
        self.acc_total = 0.0
        self.mean_class_acc.reset()
        self.mean_class_acc_total = 0.0
        self.threshold = -1
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, f1, recall, precision, auc, acc, mean_cls_acc, threshold, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.f1.add_value(f1)
        self.f1_total += f1 * mb_size
        self.recall.add_value(recall)
        self.recall_total += recall * mb_size
        self.precision.add_value(precision)
        self.precision_total += precision * mb_size
        self.auc.add_value(auc)
        self.auc_total += auc * mb_size
        self.acc.add_value(acc)
        self.acc_total += acc * mb_size
        self.mean_class_acc.add_value(mean_cls_acc)
        self.mean_class_acc_total += mean_cls_acc * mb_size
        self.num_samples += mb_size
        self.threshold = threshold

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "mean f1": self.f1.get_win_median(),
            "mean recall": self.recall.get_win_median(),
            "mean precision": self.precision.get_win_median(),
            "auc": self.auc.get_win_median(),
            "acc": self.acc.get_win_median(),
            "mean_class_acc": self.mean_class_acc.get_win_median(),
            "threshold": self.threshold
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch, log_confusion_matrix=False):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            "mean f1": self.f1_total / self.num_samples,
            "mean recall": self.recall_total / self.num_samples,
            "mean precision": self.precision_total / self.num_samples,
            "auc": self.auc_total / self.num_samples,
            "acc": self.acc_total / self.num_samples,
            "mean_class_acc": self.mean_class_acc_total / self.num_samples
        }

        if log_confusion_matrix:
            all_preds = torch.cat(self.all_preds, dim=0)
            all_labels = torch.cat(self.all_labels, dim=0)
            stats["confusion_matrix"] = metrics.conf_matrix(preds=all_preds, labels=all_labels).tolist()

        logging.log_json_stats(stats)


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)
