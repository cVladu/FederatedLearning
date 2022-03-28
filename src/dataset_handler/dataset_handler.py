import os
import pathlib
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.model.model import build_feature_extractor


def load_video(path, max_frames=None, resize=None):
    npy_path = pathlib.Path(path).with_suffix('.npy')
    if os.path.isfile(npy_path):
        if resize is None:
            raise ValueError('Resize must be specified when loading from npy')
        frames = np.load(npy_path, allow_pickle=True)
        return np.array([cv2.resize(frame, resize) for frame in frames])

    cap = cv2.VideoCapture(path)
    try:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and len(frames) >= max_frames:
                break
            if resize is not None:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
    finally:
        cap.release()
    return np.array(frames)


def video_frame_generator(path, resize=None):

    cap = cv2.VideoCapture(path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            yield frame
    finally:
        cap.release()


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 root_dir: str,
                 resize: Tuple,
                 max_frames: int,
                 batch_size: int,
                 feature_extractor: tf.keras.Model,
                 shuffle_after_epoch: bool = True,
                 partition: str = 'train'):
        self.paths = os.listdir(root_dir)
        self.paths = [os.path.join(root_dir, path) for path in self.paths]
        if partition == 'train':
            self.paths = self.paths[:int(len(self.paths) * 0.8)]
        else:
            self.paths = self.paths[int(len(self.paths) * 0.8):]
        self.path_idx = 0
        self.resize = resize
        self.max_frames = max_frames
        self.current_video = None
        self.current_video_idx = 0
        self.batch_size = batch_size
        self.idxes = None
        self.shuffle_after_epoch = shuffle_after_epoch
        self.feature_extractor = feature_extractor
        self.on_epoch_end()

    def __len__(self):
        return len(self.current_video) - 1

    def __getitem__(self, index):
        x = np.empty(shape=(self.batch_size, self.max_frames, *self.resize, 3), dtype=np.float32)
        y = np.empty(shape=(self.batch_size, *self.resize, 3), dtype=np.float32)
        tmp_idxes = self.idxes[index * self.batch_size: (index + 1) * self.batch_size]
        for batch_idx, idx in enumerate(tmp_idxes):
            x[batch_idx, ...] = self.current_video[idx: idx + self.max_frames]
            y[batch_idx, ...] = self.current_video[idx + self.max_frames]
        return x, self.feature_extractor(y)

    def on_epoch_end(self, shuffle=True):
        path_to_video = self.paths[self.path_idx]
        self.current_video = load_video(path_to_video, resize=self.resize)
        self.current_video_idx += 1 if self.current_video_idx < len(self.paths) else 0
        self.idxes = np.arange(len(self.current_video) - self.max_frames)
        if self.shuffle_after_epoch:
            np.random.shuffle(self.idxes)
        if self.current_video_idx == len(self.paths):
            if shuffle:
                np.random.shuffle(self.path_to_videos)
