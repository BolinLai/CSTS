#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""
import torch
import numpy as np


def adaptive_f1(preds, labels_hm, labels, dataset):
    """
    Automatically select the threshold getting the best f1 score.
    """
    # Numpy
    # # thresholds = np.linspace(0, 1.0, 51)
    # thresholds = np.linspace(0, 0.2, 11)
    # # thresholds = np.array([0.5])
    # preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
    # all_preds = np.zeros(shape=(thresholds.shape + labels.shape))
    # all_labels = np.zeros(shape=(thresholds.shape + labels.shape))
    # binary_labels = (labels > 0.001).astype(np.int)
    # for i in range(thresholds.shape[0]):
    #     binary_preds = (preds.squeeze(1) > thresholds[i]).astype(np.int)
    #     all_preds[i, ...] = binary_preds
    #     all_labels[i, ...] = binary_labels
    # tp = (all_preds * all_labels).sum(axis=(3, 4))
    # fg_labels = all_labels.sum(axis=(3, 4))
    # fg_preds = all_preds.sum(axis=(3, 4))
    # recall = (tp / (fg_labels + 1e-6)).mean(axis=(1, 2))
    # precision = (tp / (fg_preds + 1e-6)).mean(axis=(1, 2))
    # f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    # max_idx = np.argmax(f1)
    # return f1[max_idx], recall[max_idx], precision[max_idx], thresholds[max_idx]

    # PyTorch (To speed up calculation, use different search space for different datasets)
    if 'forecast' in dataset and 'aria' not in dataset:  # gaze forecasting on ego4d dataset
        thresholds = np.linspace(0.01, 0.07, 31)
        # thresholds = np.linspace(0, 1.0, 21)
    elif 'forecast' in dataset and 'aria' in dataset:  # gaze forecasting on aria dataset
        thresholds = np.linspace(0.0, 0.02, 21)
        # thresholds = np.linspace(0, 1.0, 21)
    else:  # gaze estimation
        thresholds = np.linspace(0, 0.02, 11)
        # thresholds = np.linspace(0, 1.0, 21)
     
    # all_preds = torch.zeros(size=(thresholds.shape + labels_hm.size()), device=labels_hm.device)
    # all_labels = torch.zeros(size=(thresholds.shape + labels_hm.size()), device=labels_hm.device)
    # binary_labels = (labels_hm > 0.001).int()  # change to 0.001
    # for i in range(thresholds.shape[0]):  # There is some space for improvement. You can calculate f1 in the loop rather than save all preds. It consumes much memory.
    #     binary_preds = (preds.squeeze(1) > thresholds[i]).int()
    #     all_preds[i, ...] = binary_preds
    #     all_labels[i, ...] = binary_labels
    # tp = (all_preds * all_labels).sum(dim=(3, 4))
    # fg_labels = all_labels.sum(dim=(3, 4))
    # fg_preds = all_preds.sum(dim=(3, 4))
    # 
    # if dataset == 'egteagaze':
    #     fixation_idx = 1
    # elif dataset in ['ego4dgaze', 'ego4dgaze_forecast', 'ego4d_av_gaze', 'ego4d_av_gaze_forecast', 'aria_gaze',
    #                  'aria_gaze_forecast', 'aria_av_gaze', 'aria_av_gaze_forecast']:
    #     fixation_idx = 0
    # else:
    #     raise NotImplementedError(f'Metrics of {dataset} is not implemented.')
    # labels_flat = labels.view(labels.size(0) * labels.size(1), labels.size(2))
    # tracked_idx = torch.where(labels_flat[:, 2] == fixation_idx)[0]
    # tp = tp.view(tp.size(0), tp.size(1)*tp.size(2)).index_select(1, tracked_idx)
    # fg_labels = fg_labels.view(fg_labels.size(0), fg_labels.size(1)*fg_labels.size(2)).index_select(1, tracked_idx)
    # fg_preds = fg_preds.view(fg_preds.size(0), fg_preds.size(1)*fg_preds.size(2)).index_select(1, tracked_idx)
    # recall = (tp / (fg_labels + 1e-6)).mean(dim=1)
    # precision = (tp / (fg_preds + 1e-6)).mean(dim=1)
    # f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    # max_idx = torch.argmax(f1)
    # 
    # return float(f1[max_idx].cpu().numpy()), float(recall[max_idx].cpu().numpy()), \
    #        float(precision[max_idx].cpu().numpy()), thresholds[max_idx]  # need np.float64 in logging rather than np.float32

    binary_labels = (labels_hm > 0.001).int()  # change to 0.001

    best_f1 = 0
    best_recall = 0
    best_precision = 0
    best_threshold = 0

    if dataset == 'egteagaze':
            fixation_idx = 1
    elif dataset in ['ego4dgaze', 'ego4dgaze_forecast', 'ego4d_av_gaze', 'ego4d_av_gaze_forecast', 'aria_gaze',
                    'aria_gaze_forecast', 'aria_av_gaze', 'aria_av_gaze_forecast']:
        fixation_idx = 0
    else:
        raise NotImplementedError(f'Metrics of {dataset} is not implemented.')

    for i in range(thresholds.shape[0]):

        binary_preds = (preds.squeeze(1) > thresholds[i]).int()

        tp = (binary_preds.unsqueeze(0) * binary_labels.unsqueeze(0)).sum(dim=(3, 4))
        fg_labels = binary_labels.unsqueeze(0).sum(dim=(3, 4))

        fg_preds = binary_preds.unsqueeze(0).sum(dim=(3, 4))
        
        labels_flat = labels.view(labels.size(0) * labels.size(1), labels.size(2))

        tracked_idx = torch.where(labels_flat[:, 2] == fixation_idx)[0]

        tp = tp.view(tp.size(0), tp.size(1)*tp.size(2)).index_select(1, tracked_idx)
        fg_labels = fg_labels.view(fg_labels.size(0), fg_labels.size(1)*fg_labels.size(2)).index_select(1, tracked_idx)
        fg_preds = fg_preds.view(fg_preds.size(0), fg_preds.size(1)*fg_preds.size(2)).index_select(1, tracked_idx)


        recall = (tp / (fg_labels + 1e-6)).mean(dim=1)
        precision = (tp / (fg_preds + 1e-6)).mean(dim=1)
        f1 = (2 * recall * precision) / (recall + precision + 1e-6)

        max_f1 = f1.max()
        if max_f1 > best_f1:
            best_f1 = max_f1
            best_recall = recall[f1.argmax()]
            best_precision = precision[f1.argmax()]
            best_threshold = thresholds[i]

    return float(best_f1.cpu().numpy()), float(best_recall.cpu().numpy()), \
        float(best_precision.cpu().numpy()), best_threshold