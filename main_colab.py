from datetime import datetime
from functools import partial

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
import cv2
import os

from tensorflow.keras import mixed_precision

from src.dataset_handler import DatasetHandlerClass
from src.model import AttentionSeekingCell, PyramidTransformer


def get_resize_func(reshape):
    def resize(x):
        frames = np.empty(shape=(x.shape[0], *reshape, 3), dtype='float16')
        for idx, frame in enumerate(x):
            frames[idx, ...] = cv2.resize(frame, reshape)
        return frames

    return resize


def resize_to_max(x, max_size):
    x = tf.image.resize(x, (max_size, max_size), preserve_aspect_ratio=False)
    return x


def lr_scheduler(epoch, lr):
    lr_list = [1e-2] * 5 + [1e-3] * 10 + [1e-4] * 5 + [1e-5] * 10

    if epoch < len(lr_list):
        return lr_list[epoch]
    else:
        return lr_list[-1]


def get_learnable_levels(epoch):
    if epoch <= 5:
        return 1
    elif epoch <= 15:
        return 2
    else:
        return 3


class LearnableLevelsCallback(tf.keras.callbacks.Callback):
    def __init__(self, func):
        super(LearnableLevelsCallback, self).__init__()
        self.func = func

    def on_epoch_begin(self, epoch, logs=None):
        self.model.set_trainable_levels(self.func(epoch))


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    dh_train = DatasetHandlerClass("/gdrive/MyDrive/Colab Notebooks/Ongoing Work/dataset/RWF-2000_npy/train")
    dh_val = DatasetHandlerClass("/gdrive/MyDrive/Colab Notebooks/Ongoing Work/dataset/RWF-2000_npy/val")
    attention_seeking_cell = AttentionSeekingCell(
        crop_size=tf.convert_to_tensor(np.array([128, 128])),
        backbone_preprocessing_fn=keras.applications.efficientnet_v2.preprocess_input,
        backbone_model=keras.applications.EfficientNetV2B0,
        lstm_units=16,
        num_heads=None,
        key_dim=None,
        pooling_layer=tf.keras.layers.GlobalAvgPool2D(),
        dropout_rate=0.4
    )
    attention_seeking_cell.build(input_shape=(1, None, None, 3))
    rnn_model = tf.keras.Sequential([
        keras.layers.Input(shape=(150, None, None, 3)),
        keras.layers.RNN(attention_seeking_cell),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32', name='classifier')
    ])
    models_path = "/gdrive/MyDrive/Colab Notebooks/Ongoing Work/models"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f'{models_path}/AttentionSeekingModel/{datetime.now().strftime("%Y%m%d-%H%M%S")}/model.{epoch:02d}.h5',
            monitor='val_loss',
            save_best_only=True, save_weights_only=True, save_format='tf'),
        keras.callbacks.TensorBoard(log_dir=f"{models_path}/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                                    write_graph=True, write_images=True, profile_batch=(2, 20)),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto'),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.LearningRateScheduler(lr_scheduler),
    ]

    rnn_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy',
                               tf.keras.metrics.AUC(name='ROC-AUC', curve='ROC'),
                               tf.keras.metrics.AUC(name='PR-AUC', curve='PR'),
                               tf.keras.metrics.TruePositives(name='TP'),
                               tf.keras.metrics.FalsePositives(name='FP'),
                               tf.keras.metrics.TrueNegatives(name='TN'),
                               tf.keras.metrics.FalseNegatives(name='FN'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               ],
                      run_eagerly=True)
    rnn_model.fit(dh_train,
                  epochs=200,
                  callbacks=callbacks,
                  validation_data=dh_val,
                  validation_freq=1,
                  workers=1,
                  use_multiprocessing=False)


def main_2():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    keras.backend.clear_session()
    dh_train = DatasetHandlerClass("/gdrive/MyDrive/Colab Notebooks/Ongoing Work/dataset/RWF-2000_npy/train",
                                   data_aug_ops=[partial(resize_to_max, max_size=128),
                                                 tf.keras.applications.mobilenet_v2.preprocess_input])
    dh_val = DatasetHandlerClass("/gdrive/MyDrive/Colab Notebooks/Ongoing Work/dataset/RWF-2000_npy/val",
                                 data_aug_ops=[partial(resize_to_max, max_size=128)])
    model = PyramidTransformer(150,
                               4,
                               1,
                               0.3,
                               backbone_model=tf.keras.applications.MobileNetV2,
                               input_shape=(128, 128, 3),
                               backbone_pooling=None,
                               final_pooling=tf.keras.layers.GlobalMaxPool2D(),
                               n_pyramid_levels=3,
                               name="PyramidTransformer")
    models_path = "/gdrive/MyDrive/Colab Notebooks/Ongoing Work/models"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f'{models_path}/PyramidTransformer/{datetime.now().strftime("%Y%m%d-%H%M%S")}/',
            monitor='val_loss',
            save_best_only=True, save_weights_only=True, save_format='tf'),
        keras.callbacks.TensorBoard(log_dir=f"{models_path}/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                                    write_graph=True, write_images=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=21, verbose=1, mode='auto'),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.LearningRateScheduler(lr_scheduler),
        LearnableLevelsCallback(get_learnable_levels),
        tf.keras.callbacks.BackupAndRestore(f"{models_path}/PyramidTransformer/backup/")
    ]

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.AUC(name='ROC-AUC', curve='ROC'),
                           tf.keras.metrics.AUC(name='PR-AUC', curve='PR'),
                           tf.keras.metrics.TruePositives(name='TP'),
                           tf.keras.metrics.FalsePositives(name='FP'),
                           tf.keras.metrics.TrueNegatives(name='TN'),
                           tf.keras.metrics.FalseNegatives(name='FN'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           ],
                  run_eagerly=True)
    model.fit(dh_train,
              epochs=200,
              callbacks=callbacks,
              validation_data=dh_val,
              validation_freq=2,
              steps_per_epoch=1000,
              validation_steps=1000,
              workers=8,
              max_queue_size=32)


if __name__ == "__main__":
    main_2()
