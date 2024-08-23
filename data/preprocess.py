import csv
import os.path

import numpy as np
import av
import subprocess
import librosa

from moviepy.editor import *
from tqdm import tqdm
from matplotlib import pyplot as plt


def trim_ego4d_videos(source_path, save_path, untrack_csv):
    """
    Trim long videos into clips.

    :param source_path: Long video path
    :param save_path:
    :param untrack_csv: Used to remove untracked frames.
    :return: None
    """
    os.makedirs(save_path, exist_ok=True)

    with open(untrack_csv, 'r') as f:
        lines = [item for item in csv.reader(f)]
    untracked = dict()
    for line in lines:
        start_hr, start_min, start_sec = line[1].split(':')
        end_hr, end_min, end_sec = line[2].split(':')
        start = int(start_hr) * 3600 + int(start_min) * 60 + int(start_sec)
        end = int(end_hr) * 3600 + int(end_min) * 60 + int(end_sec)
        if line[0] in untracked.keys():
            untracked[line[0]].append([start, end, int(line[-1])])
        else:
            untracked[line[0]] = [[start, end, int(line[-1])]]

    for item in tqdm(sorted(os.listdir(source_path))):
        if item in ['4e07da0c-450f-4c37-95e9-e793cb5d8f7f.mp4',  # We skip some videos
                    '5819e52c-4e12-4f86-ad69-76fc215dfbcb.mp4',
                    '83081c5a-8456-44d8-af67-280034f8f0a6.mp4',
                    'a77682da-cae7-4e68-8580-6cb47658b23f.mp4']:
            continue

        if os.path.splitext(item)[-1] == '.mp4':
            # loading video gfg
            video = VideoFileClip(os.path.join(source_path, item))
            duration = video.duration
            fps = video.fps

            vid = os.path.splitext(item)[0]
            os.makedirs(os.path.join(save_path, vid), exist_ok=True)

            for i in tqdm(range(0, int(duration), 5), leave=False):
                start, end = i, i + 5
                if end > duration:
                    break
                if os.path.splitext(item)[0] in untracked.keys():
                    skip = False
                    for interval in untracked[vid]:
                        if not (end < interval[0] or start > interval[1]):
                            skip = True
                            break
                    if skip:
                        continue

                clip = video.subclip(start, end)
                clip.write_videofile(os.path.join(save_path, vid, f'{vid}_t{start}_t{end}.mp4'))


def trim_aria_videos(source_path, save_path):
    """
    Trim long videos into clips.

    :param source_path: Long video path
    :param save_path:
    :return: None
    """
    os.makedirs(save_path, exist_ok=True)
    for item in tqdm(sorted(os.listdir(source_path))):
        if os.path.splitext(item)[-1] == '.mp4':
            # loading video gfg
            video = VideoFileClip(os.path.join(source_path, item))
            duration = video.duration
            fps = video.fps

            vid = os.path.splitext(item)[0]
            os.makedirs(os.path.join(save_path, vid), exist_ok=True)

            for i in tqdm(range(0, int(duration), 2), leave=False):
                start, end = i, i + 5
                if end > duration:
                    break
                clip = video.subclip(start, end)
                clip.write_videofile(os.path.join(save_path, vid, f'{vid}_t{start}_t{end-1}.mp4'))


def get_ego4d_frame_label(data_path, save_path):
    all_frames_num = 0
    all_saccade_num = 0
    all_trimmed_num = 0
    all_untracked_num = 0
    os.makedirs(save_path, exist_ok=True)
    for ann_file in os.listdir(os.path.join(data_path, 'gaze')):
        if ann_file == 'manifest.csv' or ann_file == 'manifest.ver':
            continue
        vid = ann_file.split('.')[0]
        with open(os.path.join(data_path, 'gaze', ann_file), 'r') as f:
            lines = [line for i, line in enumerate(csv.reader(f)) if i > 0]

        container = av.open(os.path.join(data_path, 'full_scale.gaze', f'{vid}.mp4'))
        fps = float(container.streams.video[0].average_rate)
        frames_length = container.streams.video[0].frames
        duration = container.streams.video[0].duration

        j = 0
        gaze_loc = list()
        for i in tqdm(range(frames_length), leave=False):
            time_stamp = i * 1 / fps  # find the accurate time stamp of each frame
            if j >= len(lines) - 2:
                break
            while float(lines[j][1]) < time_stamp:  # search the closest time of recorded location
                j += 1
            row = lines[j - 1] if abs(float(lines[j - 1][1]) - time_stamp) < abs(float(lines[j][1]) - time_stamp) else lines[j]
            x, y = float(row[5]), 1 - float(row[6])  # the initial label uses bottom-left as origin

            if i == 0:
                gaze_type = 0
            else:
                movement = np.sqrt(((x - gaze_loc[-1][1]) * 1088) ** 2 + ((y - gaze_loc[-1][2]) * 1080) ** 2)
                gaze_type = 0 if movement <= 40 else 1  # for saccade

            if not (0 <= x <= 1 and 0 <= y <= 1):
                gaze_type = 2
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
            gaze_loc.append([i, x, y, gaze_type])

        if frames_length > len(gaze_loc):
            gaze_loc.extend([[k, 0, 0, 3] for k in range(gaze_loc[-1][0]+1, frames_length)])

        all_frames_num += len(gaze_loc)
        for item in gaze_loc:
            if item[3] == 1:
                all_saccade_num += 1
            elif item[3] == 2:
                all_trimmed_num += 1
            elif item[3] == 3:
                all_untracked_num += 1

        with open(os.path.join(save_path, f'{vid}_frame_label.csv'), 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['frame', 'x', 'y', 'gaze_type'])
            csv_writer.writerows(gaze_loc)

    print('All saccade rate:', all_saccade_num / all_frames_num,
          'All trimmed rate:', all_trimmed_num / all_frames_num,
          'All untracked rate:', all_untracked_num / all_frames_num)


def get_aria_frame_label(data_path, save_path):
    all_frames_num = 0
    all_saccade_num = 0
    all_trimmed_num = 0
    all_untracked_num = 0
    os.makedirs(save_path, exist_ok=True)
    for group in os.listdir(os.path.join(data_path, 'labels')):
        with open(os.path.join(data_path, 'labels', group, 'data_summary.csv'), 'r') as f:
            meta_rows = [row for i, row in enumerate(csv.reader(f)) if i != 0]

        for i in tqdm(range(len(meta_rows))):
            subpath, video_id = meta_rows[i][6], meta_rows[i][11]
            container = av.open(os.path.join(data_path, 'full_scale', f'movie_{video_id}.mp4'))
            frame_length = container.streams.video[0].frames

            with open(os.path.join(data_path, 'labels', group, subpath), 'r') as f:
                gaze_rows = [row for i, row in enumerate(csv.reader(f)) if i != 0]

            if frame_length % 2 == 0 and frame_length != len(gaze_rows) * 2:
                print(video_id, frame_length, len(gaze_rows))
            if frame_length % 2 != 0 and len(gaze_rows) * 2 - frame_length != 1:
                print(video_id, frame_length, len(gaze_rows))

            # fps of video is 20 but it's 10 for gaze. We need to interpolate it.
            interpolate_rows = list()
            for j in range(len(gaze_rows)):
                if j != len(gaze_rows)-1:  # not the last row, interpolate using the average of two neighboring label
                    timestamp, gaze_x, gaze_y = int(gaze_rows[j][0]), float(gaze_rows[j][1]), float(gaze_rows[j][2])
                    timestamp_next, gaze_x_next, gaze_y_next = int(gaze_rows[j+1][0]), float(gaze_rows[j+1][1]), float(gaze_rows[j+1][2])
                    interpolate_rows.append([j*2, timestamp, gaze_x, gaze_y])
                    interpolate_rows.append([j*2+1, (timestamp+timestamp_next)//2, (gaze_x+gaze_x_next)/2, (gaze_y+gaze_y_next)/2])
                else:  # last row, just repeat the gaze label
                    timestamp, gaze_x, gaze_y = int(gaze_rows[j][0]), float(gaze_rows[j][1]), float(gaze_rows[j][2])
                    interpolate_rows.append([j*2, timestamp, gaze_x, gaze_y])
                    if frame_length % 2 == 0:  # don't need to repeat the last row if frame_length is odd
                        interpolate_rows.append([j*2+1, timestamp+(timestamp-int(gaze_rows[j-1][0]))//2, gaze_x, gaze_y])

            # convert gaze label to percentage and move the origin to the top-left corner
            resized_label = list()
            ori_image_edge = 1408  # the edge of original RGB frame. We have to rescale the gaze location.
            for j in range(len(interpolate_rows)):  # convert (x,y) to (1-y, x) because the origin is on top-right and xy is swapped
                resized_label.append([interpolate_rows[j][0],
                                      interpolate_rows[j][1],
                                      1 - interpolate_rows[j][3] / ori_image_edge,
                                      interpolate_rows[j][2] / ori_image_edge])

            # add gaze type
            for j in range(len(resized_label)):
                if j == 0:
                    gaze_type = 0
                else:
                    movement = np.sqrt(((resized_label[j][2] - resized_label[j-1][2]) * 640) ** 2 + ((resized_label[j][3] - resized_label[j-1][3]) * 640) ** 2)
                    gaze_type = 0 if movement <= 24 else 1  # for saccade, calculate from ego4d criterion proportional to the edge (40/1080*640)
                resized_label[j].append(gaze_type)

                if not (0 <= resized_label[j][2] <= 1 and 0 <= resized_label[j][3] <= 1):
                    gaze_type = 2  # trimmed
                    gaze_x = np.clip(resized_label[j][2], 0, 1)
                    gaze_y = np.clip(resized_label[j][3], 0, 1)
                    resized_label[j][2] = int(gaze_x)
                    resized_label[j][3] = int(gaze_y)
                    resized_label[j][4] = gaze_type

            if frame_length > len(resized_label):  # untracked
                resized_label.extend([[k, -1, 0.5, 0.5, 3] for k in range(resized_label[-1][0] + 1, frame_length)])

            os.makedirs(os.path.join(data_path, 'gaze_frame_label'), exist_ok=True)
            with open(os.path.join(data_path, 'gaze_frame_label', f'movie_{video_id}.csv'), 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['frame', 'timestamps_ns', 'x', 'y', 'gaze_type'])
                csv_writer.writerows(resized_label)

            all_frames_num += len(resized_label)
            for item in resized_label:
                if item[-1] == 1:
                    all_saccade_num += 1
                elif item[-1] == 2:
                    all_trimmed_num += 1
                elif item[-1] == 3:
                    all_untracked_num += 1

    print('All saccade rate:', all_saccade_num / all_frames_num,
          'All trimmed rate:', all_trimmed_num / all_frames_num,
          'All untracked rate:', all_untracked_num / all_frames_num)


def extract_audio(data_path, save_path, dataset):
    if dataset == 'Ego4D':
        for vid in os.listdir(data_path):
            os.makedirs(os.path.join(save_path, vid), exist_ok=True)

            for clip in os.listdir(os.path.join(data_path, vid)):
                ffmpeg_command = ['ffmpeg', '-i', os.path.join(data_path, vid, clip),
                                  '-vn', '-acodec', 'pcm_s16le', '-ac', '1',  # Ego4D has 2 identical audio channels, so convert it to mono here
                                  '-ar', '24000',  # use 24k Hz as a default
                                  os.path.join(save_path, vid, clip.replace('mp4', 'wav'))]
                subprocess.call(ffmpeg_command)

        command = f'cp {os.path.join(os.path.dirname(data_path), "missing_audio/*")} {save_path}'
        subprocess.call(command)

    elif dataset == 'Aria':
        for vid in os.listdir(data_path):
            os.makedirs(os.path.join(save_path, vid), exist_ok=True)

            for clip in tqdm(os.listdir(os.path.join(data_path, vid))):
                ffmpeg_command = ['ffmpeg', '-i', os.path.join(data_path, vid, clip),
                                  '-vn', '-acodec', 'pcm_s16le', '-ar', '24000',  # use 24k Hz as a default
                                  os.path.join(save_path, vid, clip.replace('mp4', 'wav'))]
                subprocess.call(ffmpeg_command)

    else:
        raise NotImplementedError


def audio_stft(data_path, save_path, dataset):
    window_size = 10
    step_size = 5
    eps = 1e-6

    if dataset == 'Ego4D':
        for vid in os.listdir(data_path):
            os.makedirs(os.path.join(save_path, vid), exist_ok=True)
            for clip in os.listdir(os.path.join(data_path, vid)):
                samples, sample_rate = librosa.load(os.path.join(data_path, vid, clip), sr=None, mono=False)
                nperseg = int(round(window_size * sample_rate / 1e3))
                noverlap = int(round(step_size * sample_rate / 1e3))
                spec = librosa.stft(samples, n_fft=511, window='hann', hop_length=noverlap, win_length=nperseg, pad_mode='constant')
                spec = np.log(np.real(spec * np.conj(spec)) + eps)
                np.save(os.path.join(save_path, vid, clip.replace('wav', 'npy')), spec)

    elif dataset == 'Aria':
        for vid in os.listdir(data_path):
            os.makedirs(os.path.join(save_path, vid), exist_ok=True)
            for clip in os.listdir(os.path.join(data_path, vid)):
                samples, sample_rate = librosa.load(os.path.join(data_path, vid, clip), sr=None, mono=True)  # convert to mono
                nperseg = int(round(window_size * sample_rate / 1e3))
                noverlap = int(round(step_size * sample_rate / 1e3))
                spec = librosa.stft(samples, n_fft=511, window='hann', hop_length=noverlap, win_length=nperseg, pad_mode='constant')
                spec = np.log(np.real(spec * np.conj(spec)) + eps)
                np.save(os.path.join(save_path, vid, clip.replace('wav', 'npy')), spec)

    else:
        raise NotImplementedError


def main():
    # Uncomment this code block to run preprocessing on Ego4D dataset ------------------------------------------------
    # path_to_ego4d = '/path/to/Ego4D'  # change this to your own path
    #
    # source_path = f'{path_to_ego4d}/full_scale.gaze'
    # save_path = f'{path_to_ego4d}/clips.gaze'
    # untracked_csv = f'ego4d_gaze_untracked.csv'
    # trim_ego4d_videos(source_path=source_path, save_path=save_path, untrack_csv=untracked_csv)
    #
    # data_path = path_to_ego4d
    # save_path = f'{path_to_ego4d}/gaze_frame_label'
    # get_ego4d_frame_label(data_path=data_path, save_path=save_path)
    #
    # data_path = f'{path_to_ego4d}/clips.gaze'
    # save_path = f'{path_to_ego4d}/clips.audio_24kHz'
    # extract_audio(data_path=data_path, save_path=save_path, dataset='Ego4D')
    #
    # data_path = f'{path_to_ego4d}/clips.audio_24kHz'
    # save_path = f'{path_to_ego4d}/clips.audio_24kHz_stft'
    # audio_stft(data_path, save_path, dataset='Ego4D')
    # ----------------------------------------------------------------------------------------------------------------

    # Uncomment this code block to run preprocessing on Aria dataset *************************************************
    # path_to_aria = '/path/to/Aria'
    #
    # source_path = f'{path_to_aria}/full_scale'
    # save_path = f'{path_to_aria}/clips'
    # trim_aria_videos(source_path=source_path, save_path=save_path)
    #
    # data_path = path_to_aria
    # save_path = f'{path_to_aria}/gaze_frame_label'
    # get_aria_frame_label(data_path=data_path, save_path=save_path)
    #
    # data_path = f'{path_to_aria}/clips'
    # save_path = f'{path_to_aria}/clips.audio_24kHz'
    # extract_audio(data_path=data_path, save_path=save_path, dataset='Aria')
    #
    # data_path = f'{path_to_aria}/clips.audio_24kHz'
    # save_path = f'{path_to_aria}/clips.audio_24kHz_stft'
    # audio_stft(data_path, save_path, dataset='Aria')
    # ****************************************************************************************************************

    pass


if __name__ == '__main__':
    main()
