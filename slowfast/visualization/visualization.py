import cv2
import av
import os
import torch.nn
import torch.nn.functional
import numpy as np


def vis_inference(inputs, labels, labels_hm, preds, meta, target_frames=None, output_dir=''):
    """
    Visualize the input, label and prediction of a model.

    :param inputs:
    :param labels:
    :param labels_hm:
    :param preds:
    :param meta:
    :param output_dir:
    :return:
    """
    inputs = inputs.cpu().numpy()  # N x C x T x H x W  (eg: 12 x 3 x 8 x 256 x 256)
    labels = labels.cpu().numpy()
    labels_hm = labels_hm.cpu().numpy()  # N x T x H x W
    preds = preds.squeeze(1).cpu().numpy()  # N x T x H x W
    target_frames = target_frames.cpu().numpy() if target_frames is not None else target_frames
    frame_idx = meta['index'].numpy()  # N x T

    for i in range(inputs.shape[0]):  # through N
        for j in range(inputs.shape[2]):  # through T
            frame = inputs[i, :, j, :, :].transpose(1, 2, 0)  # H x W x C
            frame = frame[:, :, ::-1]  # r, g, b --> b, g, r
            frame = (frame - frame.min()) / (frame.max() - frame .min()) * 255
            frame = frame.astype(np.uint8)  # bug of cv2

            if target_frames is not None:
                target = target_frames[i, :, j, :, :].transpose(1, 2, 0)  # H x W x C
                target = target[:, :, ::-1]  # r, g, b --> b, g, r
                target = (target - target.min()) / (target.max() - target.min()) * 255
                target = target.astype(np.uint8)  # bug of cv2

            label_frame = labels_hm[i, j, :, :]
            label_frame = cv2.resize(label_frame, dsize=frame.shape[:2], interpolation=cv2.INTER_LINEAR)
            label_frame = (label_frame - label_frame.min()) / (label_frame.max() - label_frame.min() + 1e-6) * 255
            heat_label = cv2.applyColorMap(label_frame.astype(np.uint8), cv2.COLORMAP_JET)
            # heat_label = cv2.cvtColor(heat_label, cv2.COLOR_BGR2RGB)
            if target_frames is None:
                add_label = cv2.addWeighted(frame, 0.6, heat_label, 0.4, 0)
            else:
                add_label = cv2.addWeighted(target, 0.6, heat_label, 0.4, 0)
            cv2.circle(add_label, (int(labels[i, j, 0] * inputs.shape[-1]), int(labels[i, j, 1] * inputs.shape[-2])), 5, (0, 255, 0), -1)

            pred_frame = preds[i, j, :, :]
            pred_frame = cv2.resize(pred_frame, dsize=frame.shape[:2], interpolation=cv2.INTER_LINEAR)
            pred_frame = (pred_frame - pred_frame.min()) / (pred_frame.max() - pred_frame.min() + 1e-6) * 255
            heat_pred = cv2.applyColorMap(pred_frame.astype(np.uint8), cv2.COLORMAP_JET)
            # heat_pred = cv2.cvtColor(heat_pred, cv2.COLOR_BGR2RGB)
            if target_frames is None:
                add_pred = cv2.addWeighted(frame, 0.6, heat_pred, 0.4, 0)
            else:
                add_pred = cv2.addWeighted(target, 0.6, heat_pred, 0.4, 0)
            cv2.circle(add_pred, (int(labels[i, j, 0] * inputs.shape[-1]), int(labels[i, j, 1] * inputs.shape[-2])), 5, (0, 255, 0), -1)

            concat = np.concatenate([add_pred, add_label], axis=1)
            subdir = os.path.join(output_dir, meta['path'][i].split('/')[-2], os.path.basename(meta['path'][i])[:-4])
            # if os.path.basename(subdir) == 'OP04-R04-ContinentalBreakfast-491290-493181-F011786-F011841':
            os.makedirs(subdir, exist_ok=True)
            cv2.imwrite(os.path.join(subdir, f'concat_{frame_idx[i, j]}.jpg'), concat)
            cv2.imwrite(os.path.join(subdir, f'masked_{frame_idx[i, j]}.jpg'), add_pred)
            cv2.imwrite(os.path.join(subdir, f'hm_{frame_idx[i, j]}.jpg'), heat_pred)
            cv2.imwrite(os.path.join(subdir, f'frame_{frame_idx[i, j]}.jpg'), frame)
            # save target frame
            # if target_frames is not None:
            #     cv2.imwrite(os.path.join(subdir, f'target_{frame_idx[i, j]}.jpg'), target)


def vis_video(inputs, labels, labels_hm, preds, meta, output_dir=''):
    """
    Save some frames with prediction, and then use generate_video.py to get a video.

    :param inputs:
    :param labels:
    :param labels_hm:
    :param preds:
    :param meta:
    :param output_dir:
    :return:
    """
    inputs = inputs.cpu().numpy()  # N x C x T x H x W  (eg: 12 x 3 x 8 x 256 x 256)
    labels = labels.cpu().numpy()
    labels_hm = labels_hm.cpu().numpy()  # N x T x H x W
    preds = preds.squeeze(1).cpu().numpy()  # N x T x H x W
    frame_idx = meta['index'].numpy()  # N x T

    for i in range(inputs.shape[0]):  # through N
        if os.path.basename(meta['path'][i])[:-4] not in \
                ['OP03-R05-Cheeseburger-746945-748905-F017922-F017978',
                 'OP03-R07-Pizza-223400-254800-F005286-F006191',
                 'OP05-R03-BaconAndEggs-562750-564460-F013502-F013552',
                 'OP05-R03-BaconAndEggs-51615-55250-F001230-F001334',
                 'OP06-R03-BaconAndEggs-189450-197710-F004526-F004766']:
            continue
        container = av.open(meta['path'][i])
        stream_name = {'video': 0}
        all_frames = [item.to_rgb().to_ndarray() for item in container.decode(**stream_name)]
        for j in range(inputs.shape[2]):  # through T
            for k in range(9):
                if k == 0:
                    frame = inputs[i, :, j, :, :].transpose(1, 2, 0)  # H x W x C
                else:
                    frame = all_frames[min(j * 9 + k, len(all_frames)-1)]
                    h, w = frame.shape[0], frame.shape[1]
                    frame = cv2.resize(frame[:, (w - h) // 2: (w - h) // 2 + h, :], dsize=(256, 256))
                frame = frame[:, :, ::-1]  # r, g, b --> b, g, r
                frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
                frame = frame.astype(np.uint8)  # bug of cv2

                pred_frame = preds[i, j, :, :]
                pred_frame = cv2.resize(pred_frame, dsize=frame.shape[:2], interpolation=cv2.INTER_LINEAR)
                pred_frame = (pred_frame - pred_frame.min()) / (pred_frame.max() - pred_frame.min() + 1e-6) * 255
                heat_pred = cv2.applyColorMap(pred_frame.astype(np.uint8), cv2.COLORMAP_JET)
                # heat_pred = cv2.cvtColor(heat_pred, cv2.COLOR_BGR2RGB)
                add_pred = cv2.addWeighted(frame, 0.6, heat_pred, 0.4, 0)
                cv2.circle(add_pred, (int(labels[i, j, 0] * inputs.shape[-1]), int(labels[i, j, 1] * inputs.shape[-2])), 5, (0, 255, 0), -1)

                subdir = os.path.join(output_dir, os.path.basename(meta['path'][i])[:-4])
                os.makedirs(subdir, exist_ok=True)
                cv2.imwrite(os.path.join(subdir, f'hm_{frame_idx[i, j]+k}.png'), add_pred)


def vis_video_forecasting(inputs, target_frames, labels, labels_hm, preds, meta, output_dir=''):
    inputs = inputs.cpu().numpy()  # N x C x T x H x W  (eg: 12 x 3 x 8 x 256 x 256)
    target_frames = target_frames.cpu().numpy()  # N x C x T x H x W  (eg: 12 x 3 x 8 x 256 x 256)
    labels = labels.cpu().numpy()
    labels_hm = labels_hm.cpu().numpy()  # N x T x H x W
    preds = preds.squeeze(1).cpu().numpy()  # N x T x H x W
    frame_idx = meta['index'].numpy()  # N x T

    for i in range(inputs.shape[0]):  # through N
        if os.path.basename(meta['path'][i])[:-4] not in \
                ['movie_506847100900710_t10_t14']:
            continue
        container = av.open(meta['path'][i])
        all_frames = [item.to_rgb().to_ndarray() for item in container.decode(video=0)]
        # forecast_start_idx = 86  # Ego4D
        forecast_start_idx = 60  # Aria
        for j in range(target_frames.shape[2]):  # through T
            # Ego4D
            # num_repeat = 9 if j != target_frames.shape[2]-1 else 1  # don't repeat heatmap for the last frame
            # Aria
            num_repeat = 5

            for k in range(num_repeat):
                frame = all_frames[forecast_start_idx + j * num_repeat + k]
                h, w = frame.shape[0], frame.shape[1]
                frame = cv2.resize(frame[:, (w - h) // 2: (w - h) // 2 + h, :], dsize=(256, 256))
                frame = frame[:, :, ::-1]  # r, g, b --> b, g, r
                frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
                frame = frame.astype(np.uint8)  # bug of cv2

                pred_frame = preds[i, j, :, :]
                pred_frame = cv2.resize(pred_frame, dsize=frame.shape[:2], interpolation=cv2.INTER_LINEAR)
                pred_frame = (pred_frame - pred_frame.min()) / (pred_frame.max() - pred_frame.min() + 1e-6) * 255
                heat_pred = cv2.applyColorMap(pred_frame.astype(np.uint8), cv2.COLORMAP_JET)
                add_pred = cv2.addWeighted(frame, 0.6, heat_pred, 0.4, 0)
                cv2.circle(add_pred, (int(labels[i, j, 0] * inputs.shape[-1]), int(labels[i, j, 1] * inputs.shape[-2])), 5, (0, 255, 0), -1)

                subdir = os.path.join(output_dir, os.path.basename(meta['path'][i])[:-4])
                os.makedirs(subdir, exist_ok=True)
                cv2.imwrite(os.path.join(subdir, f'hm_{frame_idx[i, j] + k}.png'), add_pred)


def vis_av_st_fusion(inputs, audio_frames, labels, target_frames, spatial_attn, temporal_attn, meta, output_dir=''):
    """
    Visualize weights in spatial fusion module.

    :param inputs:
    :param labels:
    :param glc:
    :param meta:
    :param output_dir:
    :return:
    """
    inputs = inputs.cpu().numpy()  # N x C x T x H x W  (eg: 12 x 3 x 8 x 256 x 256)
    audio_frames = audio_frames.cpu().numpy()  # N x 1 x T x H x W  (eg: 12 x 1 x 8 x 256 x 256)
    labels = labels.cpu().numpy() if labels is not None else None  # N x T x 3  (eg: 12 x 8 x 3)
    target_frames = target_frames.cpu().numpy()  # N x C x T x H x W  (eg: 12 x 3 x 8 x 256 x 256)
    frame_idx = meta['index'].numpy()

    T, H, W = 4, 8, 8
    spatial_attn = [spatial_attn[:, :, H*W*t:H*W*(t+1), T*H*W+t] for t in range(T)]
    spatial_attn = torch.stack(spatial_attn, axis=2)  # N x Head x T x HW  (eg: 12 x 8 x 4 x 64)
    spatial_attn = spatial_attn.reshape(shape=(spatial_attn.size(0), spatial_attn.size(1), T, H, W))  # N x Head x T'' x H'' x W''  (eg: 12 x 8 x 4 x 8 x 8)
    spatial_attn = torch.nn.functional.upsample(input=spatial_attn, size=inputs.shape[-3:], mode='trilinear')  # N x Head x T x H x W  (eg: 12 x 8 x 8 x 256 x 256)
    spatial_attn = spatial_attn.cpu().numpy()

    temporal_attn = temporal_attn.mean(dim=1)  # N x 2T x 2T
    temporal_attn = temporal_attn.cpu().numpy()

    for i in range(inputs.shape[0]):  # through N
        for j in range(inputs.shape[2]):  # through T
            frame = inputs[i, :, j, :, :].transpose(1, 2, 0)  # H x W x C
            frame = frame[:, :, ::-1]  # r, g, b --> b, g, r
            frame = (frame - frame .min()) / (frame .max() - frame .min()) * 255
            frame = frame.astype(np.uint8)  # bug of cv2

            # target = target_frames[i, :, j, :, :].transpose(1, 2, 0)  # H x W x C  (eg: 256 x 256 x 3)
            # target = target[:, :, ::-1]  # r, g, b --> b, g, r
            # target = (target - target.min()) / (target.max() - target.min()) * 255
            # target = target.astype(np.uint8)  # bug of cv2

            # save spatial correlation weights
            for k in range(spatial_attn.shape[1]):
                spatial_attn_frame = spatial_attn[i, k, j, :, :]
                spatial_attn_frame = (spatial_attn_frame - spatial_attn_frame.min()) / (spatial_attn_frame.max() - spatial_attn_frame.min() + 1e-6) * 255
                heat_spatial_attn = cv2.applyColorMap(spatial_attn_frame.astype(np.uint8), cv2.COLORMAP_JET)
                add_spatial_attn = cv2.addWeighted(frame, 0.6, heat_spatial_attn, 0.4, 0)
                if labels is not None:
                    cv2.circle(add_spatial_attn, (int(labels[i, j, 0] * inputs.shape[-1]), int(labels[i, j, 1] * inputs.shape[-2])), 5, (0, 255, 0), -1)

                subdir = os.path.join(output_dir, meta['path'][i].split("/")[-2], os.path.basename(meta['path'][i])[:-4])  # output_dir/video_id/clip_name
                os.makedirs(subdir, exist_ok=True)
                cv2.imwrite(os.path.join(subdir, f'spat_attn_{frame_idx[i, j]}_head_{k}.jpg'), add_spatial_attn)

        # save temporal correlation weights
        temporal_correlation = temporal_attn[i, :, :]
        subdir = os.path.join(output_dir, meta['path'][i].split("/")[-2], os.path.basename(meta['path'][i])[:-4])  # output_dir/video_id/clip_name
        os.makedirs(subdir, exist_ok=True)
        np.savetxt(os.path.join(subdir, f'temporal_attn.txt'), temporal_correlation)
