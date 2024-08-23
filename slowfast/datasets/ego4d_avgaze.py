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
from .transform import create_random_augment
from .spec_augment import combined_transforms

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ego4d_av_gaze(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Ego4d video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Ego4d video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1
        path_to_video_2
        ...
        path_to_video_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in ["train", "val", "test"], "Split '{}' not supported for Ego4d_AV_Gaze".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)

        logger.info("Constructing Eg4d_av_gaze {}...".format(mode))
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
        # path_to_file = os.path.join(
        #     self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        # )
        if self.mode == 'train':
            path_to_file = 'preprocessing/train_ego4d_gaze.csv'
        elif self.mode == 'val' or self.mode == 'test':
            path_to_file = 'preprocessing/test_ego4d_gaze.csv'
        else:
            raise ValueError(f"Don't support mode {self.mode}.")

        assert pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._path_to_audios = []
        self._full_frame_inputs = []  # used for prediction in each frame
        self._labels = dict()
        self._spatial_temporal_idx = []

        if os.path.exists(self.cfg.DATA.PATH_PREFIX):
            pass
        elif os.path.exists('/data/Ego4D/v1/clips.gaze'):
            self.cfg.DATA.PATH_PREFIX = '/data/Ego4D/v1/clips.gaze'
        else:
            self.cfg.DATA.PATH_PREFIX = '/srv/rehg-lab/flash6/blai38/Datasets/Ego4D/clips.gaze'
        with pathmgr.open(path_to_file, "r") as f:
            paths = [item for item in f.read().splitlines()]
            for clip_idx, path in enumerate(paths):
                for idx in range(self._num_clips):  # self._num_clips=1 if you don't use aggregation in test
                    self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, path))
                    self._spatial_temporal_idx.append(idx)  # used in test
                    self._video_meta[clip_idx * self._num_clips + idx] = {}  # only used in torchvision backend

                if self.cfg.TEST.FULL_FRAME_TEST is True and self.mode in ['val', 'test']:
                    # container = av.open(os.path.join(self.cfg.DATA.PATH_PREFIX, path))
                    # frames_length = container.streams.video[0].frames
                    frames_length = 150
                    indices = []
                    start_idx = 0
                    while start_idx < frames_length:
                        for i in range(start_idx, min(start_idx + self.cfg.DATA.SAMPLING_RATE+1, frames_length)):
                            idx = []
                            for j in range(self.cfg.DATA.NUM_FRAMES):
                                idx.append(min(i + j * (self.cfg.DATA.SAMPLING_RATE+1), frames_length - 1))
                            indices.append((os.path.join(self.cfg.DATA.PATH_PREFIX, path), idx))
                        start_idx = indices[-1][1][-1] + 1
                    self._full_frame_inputs.extend(indices)

        assert (len(self._path_to_videos) > 0), "Failed to load Ego4d_av_gaze split {} from {}".format(self.mode, path_to_file)

        for video_path in self._path_to_videos:
            self._path_to_audios.append(video_path.replace('clips.gaze', 'clips.gaze.audio_stft').replace('.mp4', '.npy'))

        # Read gaze label
        logger.info('Loading Gaze Labels...')
        for path in tqdm(self._path_to_videos):
            video_name = path.split('/')[-2]
            if video_name in self._labels.keys():
                pass
            else:
                label_name = video_name + '_frame_label.csv'
                prefix = os.path.dirname(self.cfg.DATA.PATH_PREFIX)
                with open(os.path.join(f'{prefix}/gaze_frame_label', label_name), 'r') as f:
                    rows = [list(map(float, row)) for i, row in enumerate(csv.reader(f)) if i > 0]
                self._labels[video_name] = np.array(rows)[:, 1:]  # [x, y, type,] in line with egtea format

        logger.info("Constructing Ego4D_AV dataloader (size: {}) from {}".format(len(self._path_to_videos), path_to_file))

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
            if self.cfg.TEST.FULL_FRAME_TEST is False:  # actually temporal_sample_index=0 regardless of FULL_FRAME_TEST
                temporal_sample_index = (self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS)  # = 0
            else:
                temporal_sample_index = 0
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            if self.cfg.TEST.FULL_FRAME_TEST is False:  # actually spatial_sample_index=1 regardless of FULL_FRAME_TEST
                spatial_sample_index = ((self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS) if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1)  # = 1
            else:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                # Don't understand why different scale is used when NUM_SPATIAL_CROPS>1
                # if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                # else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2 + [self.cfg.DATA.TEST_CROP_SIZE]
            )  # = (256, 256, 256)
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        sampling_rate = utils.get_random_sampling_rate(self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE, self.cfg.DATA.SAMPLING_RATE)
        # = 8

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatedly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            if self.mode == 'train' or self.cfg.TEST.FULL_FRAME_TEST is False:
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

                frame_length = video_container.streams.video[0].frames
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
                    get_frame_idx=True
                )

                # if self._path_to_audios[index].split('/')[-2] not in ['0d271871-c8ba-4249-9434-d39ce0060e58', '7d8b9b9f-7781-4357-a695-c88f7c7f7591']:
                audio = np.load(self._path_to_audios[index])
                audio_idx = (frames_idx / frame_length) * audio.shape[1]
                audio_idx = torch.round(audio_idx).int()
                audio_idx = torch.clip(audio_idx, 128, audio.shape[1]-1-128)
                audio_frames = np.stack([audio[:, idx-128:idx+128] for idx in audio_idx], axis=0)
                audio_frames = audio_frames[np.newaxis, ...]
                # else:  # some videos don't have audio stream
                #     audio_frames = np.full(shape=(1, frames.shape[0], 256, 256), fill_value=-13, dtype=np.float32)
                audio_frames = torch.as_tensor(audio_frames)

                # add spectrogram augmentation
                # spectrograms = list()
                # for idx in range(audio_frames.size(1)):
                #     spec = audio_frames[0, idx:idx+1, :, :]
                #     spec = combined_transforms(spec)
                #     spectrograms.append(spec)
                # audio_frames = torch.cat(spectrograms, dim=0)
                # audio_frames = audio_frames.unsqueeze(0)

                # Get gaze label on the last frame
                video_path = self._path_to_videos[index]
                video_name, clip_name = video_path.split('/')[-2:]
                clip_tstart, clip_tend = clip_name[:-4].split('_')[-2:]  # get start and end time
                clip_tstart, clip_tend = int(clip_tstart[1:]), int(clip_tend[1:])  # remove 't'
                clip_fstart, clip_fend = clip_tstart * self.cfg.DATA.TARGET_FPS, clip_tend * self.cfg.DATA.TARGET_FPS
                # frames_global_idx = frames_idx.numpy() + clip_fstart - 1  # should not -1 here
                frames_global_idx = frames_idx.numpy() + clip_fstart
                if self.mode not in ['test'] and frames_global_idx[-1] >= self._labels[video_name].shape[0]:  # Some frames don't have labels. Try to use another one
                    # logger.info('No annotations:', video_name, clip_name)
                    index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                label = self._labels[video_name][frames_global_idx, :]
                # label[:, 0][np.where(label[:, 2] == 0)] = 0.5  # In untracked frame, set gaze at the center initially. It will be covered by a uniform distribution.
                # label[:, 1][np.where(label[:, 2] == 0)] = 0.5

                # If decoding failed (wrong format, video is too short, and etc),
                # select another video.
                if frames is None:
                    logger.warning("Failed to decode video idx {} from {}; trial {}".format(index, self._path_to_videos[index], i_try))
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

            else:
                video_container = None
                video_path, frames_idx = self._full_frame_inputs[index]
                frames_idx = torch.as_tensor(np.array(frames_idx))
                try:
                    video_container = container.get_video_container(
                        video_path,
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
                        index = random.randint(0, len(self._full_frame_inputs) - 1)
                    continue

                frame_length = video_container.streams.video[0].frames
                # Decode video. Meta info is used to perform selective decoding.
                frames = [frame.to_rgb().to_ndarray() for frame in video_container.decode(video=0)]
                frames = torch.as_tensor(np.stack(frames))
                frames = frames[frames_idx, ...]

                # Load audio signal
                # if video_path.split('/')[-2] not in ['0d271871-c8ba-4249-9434-d39ce0060e58', '7d8b9b9f-7781-4357-a695-c88f7c7f7591']:
                audio = np.load(video_path.replace('clips.gaze', 'clips.gaze.audio_stft').replace('.mp4', '.npy'))
                audio_idx = (frames_idx / frame_length) * audio.shape[1]
                audio_idx = torch.round(audio_idx).int()
                audio_idx = torch.clip(audio_idx, 128, audio.shape[1]-1-128)
                audio_frames = np.stack([audio[:, idx-128:idx+128] for idx in audio_idx], axis=0)
                audio_frames = audio_frames[np.newaxis, ...]
                # else:  # some videos don't have audio stream
                #     audio_frames = np.full(shape=(1, frames.shape[0], 256, 256), fill_value=-13, dtype=np.float32)
                audio_frames = torch.as_tensor(audio_frames)

                # Get gaze label on the last frame
                video_name, clip_name = video_path.split('/')[-2:]
                clip_tstart, clip_tend = clip_name[:-4].split('_')[-2:]  # get start and end time
                clip_tstart, clip_tend = int(clip_tstart[1:]), int(clip_tend[1:])  # remove 't'
                clip_fstart, clip_fend = clip_tstart * self.cfg.DATA.TARGET_FPS, clip_tend * self.cfg.DATA.TARGET_FPS
                # frames_global_idx = frames_idx.numpy() + clip_fstart - 1  # should not -1 here
                frames_global_idx = frames_idx.numpy() + clip_fstart
                if self.mode not in ['test'] and frames_global_idx[-1] >= self._labels[video_name].shape[0]:  # Some frames don't have labels. Try to use another one
                    # logger.info('No annotations:', video_name, clip_name)
                    index = random.randint(0, len(self._full_frame_inputs) - 1)
                    continue
                label = self._labels[video_name][frames_global_idx, :]
                # label[:, 0][np.where(label[:, 2] == 0)] = 0.5  # In untracked frame, set gaze at the center initially. It will be covered by a uniform distribution.
                # label[:, 1][np.where(label[:, 2] == 0)] = 0.5

                # If decoding failed (wrong format, video is too short, and etc),
                # select another video.
                if frames is None:
                    logger.warning("Failed to decode video idx {} from {}; trial {}".format(index, self._path_to_videos[index], i_try))
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._full_frame_inputs) - 1)
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

            frames = utils.pack_pathway_output(self.cfg, frames)

            # label_hm = np.zeros(shape=(frames[0].size(1), frames[0].size(2), frames[0].size(3)))
            label_hm = np.zeros(shape=(frames[0].size(1), frames[0].size(2) // 4, frames[0].size(3) // 4))
            for i in range(label_hm.shape[0]):
                # if label[i, 2] == 0:  # if gaze is untracked, use uniform distribution
                #     label_hm[i, :, :] = label_hm[i, :, :] + 1 / (label_hm.shape[1] * label_hm.shape[2])
                # else:
                self._get_gaussian_map(label_hm[i, :, :], center=(label[i, 0] * label_hm.shape[2], label[i, 1] * label_hm.shape[1]),
                                       kernel_size=self.cfg.DATA.GAUSSIAN_KERNEL, sigma=-1)  # sigma=-1 means use default sigma
                d_sum = label_hm[i, :, :].sum()
                if d_sum == 0:  # gaze may be outside the image
                    label_hm[i, :, :] = label_hm[i, :, :] + 1 / (label_hm.shape[1] * label_hm.shape[2])
                elif d_sum != 1:  # gaze may be right at the edge of image
                    label_hm[i, :, :] = label_hm[i, :, :] / d_sum

            label_hm = torch.as_tensor(label_hm).float()
            return frames, audio_frames, label, label_hm, index, {'path': video_path, 'index': frames_global_idx}
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
