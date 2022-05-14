import math
import os
import glob
import pathlib
from typing import Tuple, List, Callable

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def load_video(video_path: str) -> np.array:
    """
    Loads a video from a given path.
    :param video_path: str. Path to the video.
    :return: np.array. A numpy array of the video.
    """
    if isinstance(video_path, tf.Tensor):
        video_path = str(video_path.numpy())
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

    finally:
        cap.release()
    return np.array(frames)


def save_npy(source_dir: str, target_dir: str):
    """
    Saves all videos in a directory to a numpy array.
    :param source_dir: str. Path to the directory containing the videos.
    :param target_dir: str. Path to the directory to save the numpy arrays.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for video_path in tqdm(glob.glob(os.path.join(source_dir, "*.avi"))):
        video = load_video(video_path)
        np.save(os.path.join(target_dir, os.path.basename(video_path)), video)


class DatasetHandlerClass(tf.keras.utils.Sequence):

    def __init__(self, dataset_path: str, data_aug_ops: List[Callable] = None, batch_size=1):
        self.video_label_list = glob.glob(os.path.join(dataset_path, "**", "*.npy"))
        self.data_aug_ops = data_aug_ops
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.video_label_list)

    def __len__(self):
        return math.ceil(len(self.video_label_list) / self.batch_size)

    def __getitem__(self, idx: int):
        video_paths = self.video_label_list[idx * self.batch_size: idx * self.batch_size + self.batch_size]
        X = []
        y = []
        for path_ in video_paths:
            video = np.load(path_)
            if self.data_aug_ops:
                for op in self.data_aug_ops:
                    video = op(video)
            X.append(video)
            y.append(pathlib.Path(path_).parent.name == "Fight")
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == "__main__":
    src_dir = "/home/cvladu/data/federated_learning/FederatedLearning/data/RWF-2000"
    dst_dir = "/home/cvladu/data/federated_learning/FederatedLearning/data/RWF-2000_npy"
    for f1 in ['train', 'val']:
        for f2 in ['Fight', 'NonFight']:
            src_path = os.path.join(src_dir, f1, f2)
            dst_path = os.path.join(dst_dir, f1, f2)
            save_npy(src_path, dst_path)
