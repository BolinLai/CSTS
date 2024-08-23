#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
from collections import deque
import torch
from fvcore.common.timer import Timer

import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc

logger = logging.get_logger(__name__)


class TestGazeMeter(object):
    """
    Measure teting stats.
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
        Construct tensors to store the predictions and labels.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video (=1 for gaze tasks).
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            dataset (str): name of the dataset.
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
        self.clip_count = torch.zeros(num_videos).long()
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
        self.recall = list()
        self.precision = list()
        self.f1 = list()

        self.preds = list()
        self.labels_hm = list()
        self.labels = list()

    def update_stats(self, f1, recall, precision, preds, labels_hm, labels):
        labels_flat = labels.view(labels.size(0) * labels.size(1), -1)
        mb_size = torch.where(labels_flat[:, 2] == 1)[0].size(0)
        self.num_samples += mb_size
        self.f1.append(f1 * mb_size)
        self.recall.append(recall * mb_size)
        self.precision.append(precision * mb_size)

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
        preds = torch.cat(self.preds, dim=0)
        labels_hm = torch.cat(self.labels_hm, dim=0)
        labels = torch.cat(self.labels, dim=0)
        f1, recall, precision, threshold = metrics.adaptive_f1(preds, labels_hm, labels, dataset=self.dataset)

        self.stats = {
            "split": "test_final",
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "threshold": threshold,
        }

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
        self.f1 = ScalarMeter(cfg.LOG_PERIOD)
        self.recall = ScalarMeter(cfg.LOG_PERIOD)
        self.recall_total = 0.0
        self.precision = ScalarMeter(cfg.LOG_PERIOD)
        self.precision_total = 0.0
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
        self.recall.reset()
        self.recall_total = 0.0
        self.precision.reset()
        self.precision_total = 0.0
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

    def update_stats(self, f1, recall, precision, threshold, loss, lr, mb_size):
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
        self.f1.add_value(f1)
        self.recall.add_value(recall)
        self.recall_total += recall * mb_size
        self.precision.add_value(precision)
        self.precision_total += precision * mb_size
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
            "f1": self.f1.get_win_median(),
            "recall": self.recall.get_win_median(),
            "precision": self.precision.get_win_median(),
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
        }
        recall = self.recall_total / self.num_samples
        precision = self.precision_total / self.num_samples
        f1 = 2 * recall * precision / (recall + precision + 1e-6)
        stats['f1'] = f1
        stats['recall'] = recall
        stats['precision'] = precision

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
        self.f1 = ScalarMeter(cfg.LOG_PERIOD)
        self.recall = ScalarMeter(cfg.LOG_PERIOD)
        self.recall_total = 0.0
        self.precision = ScalarMeter(cfg.LOG_PERIOD)
        self.precision_total = 0.0
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
        self.recall.reset()
        self.recall_total = 0.0
        self.precision.reset()
        self.precision_total = 0.0
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

    def update_stats(self, f1, recall, precision, labels, threshold):
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
        self.f1.add_value(f1)
        self.recall.add_value(recall)
        self.recall_total += recall * mb_size
        self.precision.add_value(precision)
        self.precision_total += precision * mb_size
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
            "f1": self.f1.get_win_median(),
            "recall": self.recall.get_win_median(),
            "precision": self.precision.get_win_median(),
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
        }
        recall = self.recall_total / self.num_samples
        precision = self.precision_total / self.num_samples
        f1 = 2 * recall * precision / (recall + precision + 1e-6)
        stats['recall'] = recall
        stats['precision'] = precision
        stats['f1'] = f1

        logging.log_json_stats(stats)


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
