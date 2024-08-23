#!/usr/bin/env python3

import numpy as np
import os
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.utils.metrics as metrics
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import TestGazeMeter
from slowfast.utils.utils import frame_softmax
# from slowfast.visualization.visualization import vis_inference, vis_video_forecasting, vis_av_st_fusion

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestGazeMeter): testing meters to log and ensemble the testing results.
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, audio_frames, labels, labels_hm, video_idx, meta) in enumerate(test_loader):
    # for cur_iter, (inputs, audio_frames, labels, labels_hm, target_frames, video_idx, meta) in enumerate(test_loader):  # return target frames for forecasting
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            audio_frames = audio_frames.cuda()
            labels = labels.cuda()
            labels_hm = labels_hm.cuda()
            video_idx = video_idx.cuda()
        test_meter.data_toc()

        # Perform the forward pass.
        preds = model(inputs, audio_frames)

        preds = frame_softmax(preds, temperature=2)  # KLDiv

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, labels_hm, video_idx = du.all_gather([preds, labels, labels_hm, video_idx])

        # Compute the metrics.
        if cfg.NUM_GPUS:  # compute on cpu
            preds = preds.cpu()
            labels = labels.cpu()
            labels_hm = labels_hm.cpu()
            video_idx = video_idx.cpu()

        preds_rescale = preds.detach().view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
        preds_rescale = (preds_rescale - preds_rescale.min(dim=-1, keepdim=True)[0]) / (preds_rescale.max(dim=-1, keepdim=True)[0] - preds_rescale.min(dim=-1, keepdim=True)[0] + 1e-6)
        preds_rescale = preds_rescale.view(preds.size())
        f1, recall, precision, threshold = metrics.adaptive_f1(preds_rescale, labels_hm, labels, dataset=cfg.TEST.DATASET)

        # Visualization
        # gaze forecast
        # if cur_iter <= 10:
        # vis_inference(inputs=inputs[0], labels=labels, labels_hm=labels_hm, preds=preds_rescale, target_frames=target_frames,
        #               meta=meta, output_dir=os.path.join(cfg.OUTPUT_DIR, 'visualization'))
        # vis_video_forecasting(inputs=inputs[0], labels=labels, labels_hm=labels_hm, preds=preds_rescale, target_frames=target_frames,
        #                       meta=meta, output_dir=os.path.join(cfg.OUTPUT_DIR, 'frames_for_video'))
        # vis_av_st_fusion(inputs=inputs[0], audio_frames=audio_frames, labels=labels, target_frames=target_frames,
        #                  spatial_attn=spatial_attn, temporal_attn=temporal_attn, meta=meta,
        #                  output_dir=os.path.join(cfg.OUTPUT_DIR, 'st_visualization'))

        test_meter.iter_toc()

        # Update and log stats.
        test_meter.update_stats(f1, recall, precision, preds=preds_rescale, labels_hm=labels_hm, labels=labels)  # If running  on CPU (cfg.NUM_GPUS == 0), use 1 to represent 1 CPU.
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform testing on the video model.
    Args:
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (test_loader.dataset.num_videos % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS) == 0)
    # Create meters for multi-view testing.
    test_meter = TestGazeMeter(
        num_videos=test_loader.dataset.num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        num_cls=cfg.MODEL.NUM_CLASSES,
        overall_iters=len(test_loader),
        dataset=cfg.TEST.DATASET
    )

    writer = None  # Forbid use tensorboard for test

    # Perform testing on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)

    logger.info("Testing finished!")
