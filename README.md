# Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation

### ECCV 2024

### [Project Page](https://bolinlai.github.io/CSTS-EgoGazeAnticipation/) | [Paper](https://arxiv.org/pdf/2305.03907)



### TODO:

- [ ] Codes
- [ ] Checkpoints
- [x] Data Split
- [ ] Update README (in progress)

## Contents
- [Problem Definition](#problem-definition)
- [Approach](#approach)
- [Setup](#setup)
- [Datasets](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [BibTeX](#bibtex)
- [Acknowledge](#acknowledgement)


## Problem Definition

 <img src='https://bolinlai.github.io/CSTS-EgoGazeAnticipation/figures/teaser.png'/>


## Approach

 <img src='https://bolinlai.github.io/CSTS-EgoGazeAnticipation/figures/method.png'/>


## Setup

Set up the virtual environment using following commands.

```shell
conda env create -f virtual_env.yaml
conda activate csts
python setup.py build develop
```

## Dataset

### Ego4D

We use the same split in our [prior work](https://github.com/BolinLai/GLC/blob/main/slowfast/datasets/DATASET.md) which is also recommended by the [official website](https://ego4d-data.org/docs/data/gaze/). Please follow the steps below to prepare the dataset. 

1. Read the agreement and apply for Ego4D access on the [website](https://ego4d-data.org/docs/start-here/) (It may take a few days to get approved). Follow the instructions in "Download The CLI" and this [guidance](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md) to set up the CLI tool.


2. We only need to download the subset of Ego4D that has gaze data. You can download all the gaze data using the CLI tool on this [page](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md).


3. Gaze annotations are organized in a bunch of csv files. Each file corresponds to a video. Unfortunately, Ego4D hasn't provided a command to download all of these videos yet. You need to download videos via the video ids (i.e. the name of each csv file) using the CLI tool and --video_uids following [instructions](https://ego4d-data.org/docs/CLI/).


4. Two videos in Ego4D subset don't have audio streams. We provide the missing audio files [here](https://drive.google.com/drive/folders/1iZuuRiflog9AazCtLXa9PbIYg-S3vENs?usp=drive_link).


5. Please reorganize the video clips and annotations in this structure:

    ```
    Ego4D
    |- full_scale.gaze
    |  |- 0d271871-c8ba-4249-9434-d39ce0060e58.mp4
    |  |- 1e83c2d1-ff03-4181-9ab5-a3e396f54a93.mp4
    |  |- 2bb31b69-fcda-4f54-8338-f590944df999.mp4
    |  |- ...
    |
    |- gaze
    |  |- 0d271871-c8ba-4249-9434-d39ce0060e58.csv
    |  |- 1e83c2d1-ff03-4181-9ab5-a3e396f54a93.csv
    |  |- 2bb31b69-fcda-4f54-8338-f590944df999.csv
    |  |- ...
    |
    |- missing_audio
       |- 0d271871-c8ba-4249-9434-d39ce0060e58.wav
       |- 7d8b9b9f-7781-4357-a695-c88f7c7f7591.wav
    ```

6. Uncomment the Ego4D code block in `data/preprocess.py: main()` and update the variable `path_to_ego4d` to your local path of Ego4D dataset. Then run the command.

    ```shell
    python preprocess.py
    ```
   The data after preprocessing is saved in the following directories.

   `clips.gaze`: The video clips of 5s duration.

   `gaze_frame_label`: The gaze target location in each video frame.

   `clips.audio_24kHz`: The audio streams resampled in 24kHz.

   `clips.audio_24kHz_stft`: The audio streams after STFT.

### Aria

TODO

(The Aria dataset is in a very different format than it was when we started our work. We need more time to update our codes. Thank you for your patience.)


## Training

We use MViT as our backbone. The Kinetics-400 pre-trained model is released [here](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md) (i.e., Kinetics/MVIT_B_16x4_CONV). We find this checkpoint is no longer available on that page. We thus provide the pretrained weights via this [link](https://drive.google.com/file/d/1cZjY9jK7urPxvZfYumIVVVvdXLmVsiJk/view?usp=drive_link).

Train on Ego4D dataset.

```shell
CUDA_VISIBLE_DEVICES=0,1 python tools/run_net.py \
    --init_method tcp://localhost:9880 \
    --cfg configs/Ego4d/CSTS_Ego4D_Gaze_Forecast.yaml \
    TRAIN.BATCH_SIZE 8 \
    TEST.BATCH_SIZE 96 \
    NUM_GPUS 2 \
    TRAIN.CHECKPOINT_FILE_PATH /path/to/pretrained/K400_MVIT_B_16x4_CONV.pyth \
    OUTPUT_DIR out/csts_ego4d \
    DATA.PATH_PREFIX /path/to/Ego4D/clips.gaze \
    MODEL.LOSS_FUNC kldiv+egonce \
    MODEL.LOSS_ALPHA 0.05 \
    RNG_SEED 21
```

Train on Aria dataset.

```shell
CUDA_VISIBLE_DEVICES=0,1 python tools/run_net.py \
    --init_method tcp://localhost:9880 \
    --cfg configs/Aria/CSTS_Aria_Gaze_Forecast.yaml \
    TRAIN.BATCH_SIZE 8 \
    TEST.BATCH_SIZE 96 \
    NUM_GPUS 2 \
    TRAIN.CHECKPOINT_FILE_PATH /path/to/pretrained/K400_MVIT_B_16x4_CONV.pyth \
    OUTPUT_DIR out/csts_aria \
    DATA.PATH_PREFIX /path/to/Aria/clips \
    MODEL.LOSS_FUNC kldiv+egonce \
    MODEL.LOSS_ALPHA 0.05 \
    RNG_SEED 21
```

Note: You need to replace `TRAIN.CHECKPOINT_FILE_PATH` with your local path to pretrained MViT checkpoint, and replace `DATA.PATH_PREFIX` with your local path to video clips.

The checkpoints are saved in `./out` directory.


### Evaluation

Run evaluation on Ego4D dataset.

```shell
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
    --cfg configs/Ego4d/CSTS_Ego4D_Gaze_Forecast.yaml \
    TRAIN.ENABLE False \
    TEST.BATCH_SIZE 24 \
    NUM_GPUS 1 \
    OUTPUT_DIR out/csts_ego4d/checkpoints/test_epoch5 \
    TEST.CHECKPOINT_FILE_PATH out/csts_ego4d/checkpoints/checkpoint_epoch_00005.pyth \
    [DATA_LOADER.RETURN_TARGET_FRAME True]
```

Run evaluation on Aria dataset.

```shell
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
    --cfg configs/Aria/CSTS_Aria_Gaze_Forecast.yaml \
    TRAIN.ENABLE False \
    TEST.BATCH_SIZE 24 \
    NUM_GPUS 1 \
    OUTPUT_DIR out/csts_aria/checkpoints/test_epoch5 \
    TEST.CHECKPOINT_FILE_PATH out/csts_aria/checkpoints/checkpoint_epoch_00005.pyth \
    [DATA_LOADER.RETURN_TARGET_FRAME True]
```

Note: You need to replace `OUTPUT_DIR` with the path of saving evaluation logs, and replace `TEST.CHECKPOINT_FILE_PATH` with the path of the checkpoint that you want to evaluate.
