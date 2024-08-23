#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random

import av
import cv2
import numpy as np
import csv
import torch
import torch.utils.data
from torchvision import transforms
from tqdm import tqdm

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment, color_jitter

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Aria_av_gaze_forecast(torch.utils.data.Dataset):
    """
    Aria video loader. Construct the Aria video loader, then sample clips from the videos.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Aria video loader with a given csv file. The format of the csv file is:
        ```
        path_to_video_1
        path_to_video_2
        ...
        path_to_video_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in ["train", "val", "test"], "Split '{}' not supported for Aria_AV_Gaze".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)

        logger.info("Constructing Aria_av_gaze {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

        if self.mode == "train" and self.cfg.AUG.ENABLE:  # use RandAug
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video and audio loader.
        """
        if self.mode == 'train':
            path_to_file = 'data/train_aria_gaze.csv'
        elif self.mode == 'val' or self.mode == 'test':
            path_to_file = 'data/test_aria_gaze.csv'
        else:
            raise ValueError(f"Don't support mode {self.mode}.")

        assert pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._path_to_audios = []
        self._labels = dict()
        self._spatial_temporal_idx = []

        with pathmgr.open(path_to_file, "r") as f:
            paths = [item for item in f.read().splitlines()]
            for clip_idx, path in enumerate(paths):
                for idx in range(self._num_clips):  # self._num_clips=1 if you don't use aggregation in test
                    self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, path))
                    self._spatial_temporal_idx.append(idx)  # used in test
                    self._video_meta[clip_idx * self._num_clips + idx] = {}  # only used in torchvision backend
        assert (len(self._path_to_videos) > 0), "Failed to load Aria_av_gaze split {} from {}".format(self.mode, path_to_file)

        for video_path in self._path_to_videos:
            self._path_to_audios.append(video_path.replace('clips', 'clips.audio_24kHz_stft').replace('.mp4', '.npy'))

        # Read gaze label
        logger.info('Loading Gaze Labels...')
        for path in tqdm(self._path_to_videos):
            video_name = path.split('/')[-2]
            if video_name in self._labels.keys():
                pass
            else:
                label_name = video_name + '.csv'
                prefix = os.path.dirname(self.cfg.DATA.PATH_PREFIX)
                with open(os.path.join(f'{prefix}/gaze_frame_label', label_name), 'r') as f:
                    rows = [list(map(float, row)) for i, row in enumerate(csv.reader(f)) if i > 0]
                self._labels[video_name] = np.array(rows)[:, 2:]  # [x, y, type,] in line with egtea format

        logger.info("Constructing Aria_AV dataloader (size: {}) from {}".format(len(self._path_to_videos), path_to_file))

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]  # 256
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]  # 320
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE  # 256
            if short_cycle_idx in [0, 1]:
                crop_size = int(round(self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx] * self.cfg.MULTIGRID.DEFAULT_S))
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S))

        elif self.mode in ["val", "test"]:
            temporal_sample_index = 1  # temporal_sample_index is in [0, 1]. Corresponding to left and right.
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = ((self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS) if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1)  # = 1
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
            )  # = (256, 256, 256)
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        sampling_rate = utils.get_random_sampling_rate(self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE, self.cfg.DATA.SAMPLING_RATE)  # = 4

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatedly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info("Failed to load video from {} with error {}".format(self._path_to_videos[index], e))

            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning("Failed to meta load video idx {} from {}; trial {}".format(index, self._path_to_videos[index], i_try))
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            ori_frame_length = video_container.streams.video[0].frames
            frame_length = 60  # only sample in the first 60 frames for gaze forecasting
            # Decode video. Meta info is used to perform selective decoding.
            frames, frames_idx = decoder.decode(
                container=video_container,
                sampling_rate=sampling_rate,
                num_frames=self.cfg.DATA.NUM_FRAMES,
                clip_idx=temporal_sample_index,
                num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,  # only used in torchvision backend
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                get_frame_idx=True,
                frames_length_limit=frame_length,  # sample in the first 90 frames
            )

            audio = np.load(self._path_to_audios[index])
            audio = audio[:, :int(audio.shape[1] * frame_length/ori_frame_length)]  # trim spectrogram to avoid using the information of last 2 seconds
            audio_idx = (frames_idx / frame_length) * audio.shape[1]
            audio_idx = torch.round(audio_idx).int()
            audio_idx = torch.clip(audio_idx, 128, audio.shape[1]-1-128)
            audio_frames = np.stack([audio[:, idx-128:idx+128] for idx in audio_idx], axis=0)
            audio_frames = audio_frames[np.newaxis, ...]
            audio_frames = torch.as_tensor(audio_frames)

            # Get gaze label
            video_path = self._path_to_videos[index]
            video_name, clip_name = video_path.split('/')[-2:]
            clip_tstart, clip_tend = clip_name[:-4].split('_')[-2:]  # get start and end time
            clip_tstart, clip_tend = int(clip_tstart[1:]), int(clip_tend[1:])  # remove 't'
            clip_fstart, clip_fend = clip_tstart * self.cfg.DATA.TARGET_FPS, clip_tend * self.cfg.DATA.TARGET_FPS
            frames_global_idx = frames_idx.numpy() + clip_fstart
            if self.mode == 'train':
                last_idx = frames_idx.numpy()[-1]
                labels_idx = np.arange(last_idx + 1 + self.cfg.DATA.SAMPLING_RATE,
                                       last_idx + 1 + ori_frame_length - frame_length)
            else:
                labels_idx = np.arange(frame_length+self.cfg.DATA.SAMPLING_RATE, ori_frame_length)  # array([64, 69, ..., 94, 99])
            labels_idx = np.linspace(labels_idx[0], labels_idx[-1], self.cfg.DATA.NUM_FRAMES).astype('int64')
            labels_global_index = labels_idx + clip_fstart
            if self.mode not in ['test'] and labels_global_index[-1] >= self._labels[video_name].shape[0]:  # Some frames don't have labels. Try to use another one
                # logger.info('No annotations:', video_name, clip_name)
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue
            label = self._labels[video_name][labels_global_index, :]

            # Get target frames where gaze labels are located (Copied from pyav_decode() in decoder.py)
            target_frames = None
            if self.cfg.DATA_LOADER.RETURN_TARGET_FRAME:
                video_container = container.get_video_container(  # Note: you have to load video twice. The old container can't be used after iteration
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
                timebase = video_container.streams.video[0].duration / ori_frame_length  # = 512.0
                video_start_pts = int(labels_idx[0] * timebase)  # start and end timestamp
                video_end_pts = int(labels_idx[-1] * timebase)
                target_video_frames, max_pts = decoder.pyav_decode_stream(
                    container=video_container,
                    start_pts=video_start_pts,
                    end_pts=video_end_pts,
                    stream=video_container.streams.video[0],
                    stream_name={"video": 0}
                )
                video_container.close()
                target_frames = [tframe.to_rgb().to_ndarray() for tframe in target_video_frames]
                target_frames = torch.as_tensor(np.stack(target_frames))
                target_frames = decoder.temporal_sampling(target_frames, 0, target_frames.size(0)-1, self.cfg.DATA.NUM_FRAMES)
                target_frames = target_frames.permute(3, 0, 1, 2)

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning("Failed to decode video idx {} from {}; trial {}".format(index, self._path_to_videos[index], i_try))
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if self.aug:
                if self.cfg.AUG.NUM_SAMPLE > 1:

                    frame_list = []
                    label_list = []
                    index_list = []
                    for _ in range(self.cfg.AUG.NUM_SAMPLE):
                        new_frames = self._aug_frame(frames, spatial_sample_index, min_scale, max_scale, crop_size)
                        label = self._labels[index]
                        new_frames = utils.pack_pathway_output(self.cfg, new_frames)
                        frame_list.append(new_frames)
                        label_list.append(label)
                        index_list.append(index)
                    return frame_list, label_list, index_list, {}

                else:
                    frames = self._aug_frame(frames, spatial_sample_index, min_scale, max_scale, crop_size)

            else:
                frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)

                if target_frames is not None:
                    frames = torch.cat([frames, target_frames], dim=1)

                # Perform data augmentation.
                frames, label = utils.spatial_sampling(
                    frames,
                    gaze_loc=label,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )

                if target_frames is not None:
                    frames, target_frames = frames[:, :frames.size(1)//2, :, :], frames[:, frames.size(1)//2:, :, :]

            frames = utils.pack_pathway_output(self.cfg, frames)

            label_hm = np.zeros(shape=(frames[0].size(1), frames[0].size(2) // 4, frames[0].size(3) // 4))
            for i in range(label_hm.shape[0]):
                self._get_gaussian_map(label_hm[i, :, :], center=(label[i, 0] * label_hm.shape[2], label[i, 1] * label_hm.shape[1]),
                                       kernel_size=self.cfg.DATA.GAUSSIAN_KERNEL, sigma=-1)  # sigma=-1 means use default sigma
                d_sum = label_hm[i, :, :].sum()
                if d_sum == 0:  # gaze may be outside the image
                    label_hm[i, :, :] = label_hm[i, :, :] + 1 / (label_hm.shape[1] * label_hm.shape[2])
                elif d_sum != 1:  # gaze may be right at the edge of image
                    label_hm[i, :, :] = label_hm[i, :, :] / d_sum

            label_hm = torch.as_tensor(label_hm).float()

            if target_frames is not None:  # if target_frames are defined, return target frames for visualization
                return frames, audio_frames, label, label_hm, target_frames, index, \
                    {'path': video_path, 'index': frames_global_idx, 'labels_index': labels_global_index}
            else:
                return frames, audio_frames, label, label_hm, index, \
                    {'path': video_path, 'index': frames_global_idx, 'labels_index': labels_global_index}
        else:
            raise RuntimeError("Failed to fetch video after {} retries.".format(self._num_retries))

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE, self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE)
        relative_scales = (None if (self.mode not in ["train"] or len(scl) == 0) else scl)
        relative_aspect = (None if (self.mode not in ["train"] or len(asp) == 0) else asp)
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    @staticmethod
    def _get_gaussian_map(heatmap, center, kernel_size, sigma):
        h, w = heatmap.shape
        mu_x, mu_y = round(center[0]), round(center[1])
        left = max(mu_x - (kernel_size - 1) // 2, 0)
        right = min(mu_x + (kernel_size - 1) // 2, w-1)
        top = max(mu_y - (kernel_size - 1) // 2, 0)
        bottom = min(mu_y + (kernel_size - 1) // 2, h-1)

        if left >= right or top >= bottom:
            pass
        else:
            kernel_1d = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma, ktype=cv2.CV_32F)
            kernel_2d = kernel_1d * kernel_1d.T
            k_left = (kernel_size - 1) // 2 - mu_x + left
            k_right = (kernel_size - 1) // 2 + right - mu_x
            k_top = (kernel_size - 1) // 2 - mu_y + top
            k_bottom = (kernel_size - 1) // 2 + bottom - mu_y

            heatmap[top:bottom+1, left:right+1] = kernel_2d[k_top:k_bottom+1, k_left:k_right+1]

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos if self.cfg.TEST.FULL_FRAME_TEST is False else self.num_full_frame_inputs

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    @property
    def num_full_frame_inputs(self):
        return len(self._full_frame_inputs)
