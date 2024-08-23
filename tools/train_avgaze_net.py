#!/usr/bin/env python3

import random
import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import EpochTimer, TrainGazeMeter, ValGazeMeter, TrainMeter, ValMeter
from slowfast.utils.utils import frame_softmax, sim_matrix

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (inputs, audio_frames, labels, labels_hm, _, meta) in enumerate(train_loader):
    # for cur_iter, (inputs, audio_frames, labels, labels_hm, target_frames, _, meta) in enumerate(train_loader):  # return target frames for forecasting
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            audio_frames = audio_frames.cuda()
            labels = labels.cuda()
            labels_hm = labels_hm.cuda()

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            if cfg.MODEL.LOSS_FUNC == 'kldiv+egonce':
                preds = model(inputs, audio_frames, return_embed=True)
            else:
                preds = model(inputs, audio_frames)

            if cfg.MODEL.LOSS_FUNC == 'kldiv+egonce':
                kldiv_fun = losses.get_loss_func('kldiv')
                egonce_fun = losses.get_loss_func('egonce')
                kldiv_fun = kldiv_fun()
                egonce_fun = egonce_fun()
                preds, v_embed, a_embed = preds
                if cfg.NUM_GPUS > 1:
                    v_embed, a_embed = du.all_gather_with_grad([v_embed, a_embed])
                preds = frame_softmax(preds, temperature=2)
                similarity = sim_matrix(v_embed, a_embed)
                kldiv_loss = kldiv_fun(preds, labels_hm)
                egonce_loss = egonce_fun(similarity)
                loss = kldiv_loss + cfg.MODEL.LOSS_ALPHA * egonce_loss
            else:
                loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)
                loss_fun = loss_fun(reduction='mean')
                loss = loss_fun(preds, labels_hm)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL)
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM)
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            loss = du.all_reduce([loss])[0]  # average across all processes
            preds, labels_hm, labels = du.all_gather([preds, labels_hm, labels])  # gather (concatenate) across all processes

            if cfg.MODEL.LOSS_FUNC == 'kldiv+egonce':
                kldiv_loss, egonce_loss = du.all_reduce([kldiv_loss, egonce_loss])
                kldiv_loss, egonce_loss = kldiv_loss.item(), egonce_loss.item()
                if (cur_iter + 1) % cfg.LOG_PERIOD == 0:
                    logger.info(f'Iter {cur_iter + 1}: kld_loss {round(kldiv_loss, 4)}, egonce_loss {round(egonce_loss, 4)}, loss {round(loss.item(), 4)}')

        loss = loss.item()

        # Compute the metrics.
        preds_rescale = preds.detach().view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
        preds_rescale = (preds_rescale - preds_rescale.min(dim=-1, keepdim=True)[0]) / (preds_rescale.max(dim=-1, keepdim=True)[0] - preds_rescale.min(dim=-1, keepdim=True)[0] + 1e-6)
        preds_rescale = preds_rescale.view(preds.size())
        f1, recall, precision, threshold = metrics.adaptive_f1(preds_rescale, labels_hm, labels, dataset=cfg.TRAIN.DATASET)
        iou = metrics.gaze_iou(preds_rescale, labels_hm, threshold=threshold)
        auc = metrics.auc(preds_rescale, labels_hm, labels, dataset=cfg.TRAIN.DATASET)

        # Update and log stats.
        train_meter.update_stats(iou, f1, recall, precision, auc, threshold, loss, lr, mb_size=inputs[0].size(0) * max(cfg.NUM_GPUS, 1))  # If running  on CPU (cfg.NUM_GPUS == 0), use 1 to represent 1 CPU.

        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/loss": loss,
                    "Train/lr": lr,
                    "Train/IoU": iou,
                    "Train/F1": f1,
                    "Train/Recall": recall,
                    "Train/Precision": precision,
                    "Train/AUC": auc
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

            if cfg.MODEL.LOSS_FUNC == 'kldiv+egonce':
                writer.add_scalars({"Train/kldiv_loss": kldiv_loss, "Train/nce_loss": egonce_loss}, global_step=data_size * cur_epoch + cur_iter,)

        train_meter.iter_toc()  # measure all reduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, audio_frames, labels, labels_hm, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            audio_frames = audio_frames.cuda()
            labels = labels.cuda()
            labels_hm = labels_hm.cuda()

        val_meter.data_toc()

        preds = model(inputs, audio_frames)
        preds = frame_softmax(preds, temperature=2)  # KLDiv

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels_hm, labels = du.all_gather([preds, labels_hm, labels])

        # Compute the metrics.
        preds_rescale = preds.detach().view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
        preds_rescale = (preds_rescale - preds_rescale.min(dim=-1, keepdim=True)[0]) / (preds_rescale.max(dim=-1, keepdim=True)[0] - preds_rescale.min(dim=-1, keepdim=True)[0] + 1e-6)
        preds_rescale = preds_rescale.view(preds.size())
        f1, recall, precision, threshold = metrics.adaptive_f1(preds_rescale, labels_hm, labels, dataset=cfg.TRAIN.DATASET)
        iou = metrics.gaze_iou(preds_rescale, labels_hm, threshold=threshold)
        auc = metrics.auc(preds_rescale, labels_hm, labels, dataset=cfg.TRAIN.DATASET)

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(iou, f1, recall, precision, auc, labels, threshold)  # If running  on CPU (cfg.NUM_GPUS == 0), use 1 to represent 1 CPU.

        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars({
                "Val/IoU": iou,
                "Val/F1": f1,
                "Val/Recall": recall,
                "Val/Precision": precision,
                "Val/AUC": auc
            }, global_step=len(val_loader) * cur_epoch + cur_iter)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    if cfg.NUM_GPUS > 0:
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (loader.construct_loader(cfg, "train", is_precise_bn=True) if cfg.BN.USE_PRECISE_STATS else None)

    # Create meters.
    train_meter = TrainGazeMeter(len(train_loader), cfg)
    val_meter = ValGazeMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)  # Seems not work when GPU=1

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_meter=train_meter,
            cur_epoch=cur_epoch,
            cfg=cfg,
            writer=writer,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(cfg, cur_epoch, None)
        is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch, None)

        # Compute precise BN stats.
        if ((is_checkp_epoch or is_eval_epoch) and cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0):
            calculate_and_update_precise_bn(
                loader=precise_bn_loader,
                model=model,
                num_iters=min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                use_gpu=cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)  # seems no influence

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                path_to_job=cfg.OUTPUT_DIR,
                model=model,
                optimizer=optimizer,
                epoch=cur_epoch,
                cfg=cfg,
                scaler=scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()

    logger.info("Training finished!")
