import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def main():
    cap = cv2.VideoCapture('dataset/RWF-2000/RWF-2000/train/Fight/0_DzLlklZa0_3.avi')

    if not cap.isOpened():
        print("Error opening video stream or file")

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.resize(frame, (224, 224)))
        else:
            break
    cap.release()
    X = np.array(frames[0]) / 255.0
    X = X[np.newaxis, ...]
    y = np.array(frames[1]) / 255.0
    y = y[np.newaxis, ...]
    out = keras.layers.Attention(use_scale=True)([X, y])
    pass


if __name__ == "__main__":
    main()