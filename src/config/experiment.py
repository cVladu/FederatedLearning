import os.path
from typing import Tuple, List
import tensorflow as tf
from tensorflow import keras

import src.dataset_handler.dataset_handler as dh
import src.model.model as m
# Parameters
# -----------------------------------------------------------------------------
RESIZE: Tuple[int, int] = (244, 244)
N_CHANNELS: int = 3
ROOT_DIR: os.path = os.path.abspath("dataset/UCF-Crime/Training-Normal-Videos-Part-1")
MAX_FRAMES: int = 20

FEATURE_EXTRACTOR: str = "MobileNetV2"
AUTOENCODER_UNITS: List[int] = [512, 256, 128]
AUTOENCODER_MASK: List[bool] = [True, True, True]

EPOCHS = 10

log_path = "logs/experiment_1"

def lr_scheduler(epoch, lr):
    lr_list = [1e-2] * 40 + [1e-3] * 50 + [1e-4] * 100 + [1e-5] * 10

    return lr_list[epoch]

def main():
    built_feature_extractor = m.build_feature_extractor(FEATURE_EXTRACTOR, input_shape=(*RESIZE, N_CHANNELS))
    train_generator = dh.DataGenerator(root_dir=ROOT_DIR,
                                       max_frames=MAX_FRAMES,
                                       resize=RESIZE,
                                       batch_size=1,
                                       feature_extractor=built_feature_extractor,
                                       shuffle_after_epoch=True)
    model_instance = m.ULSTMModel(feature_extractor=FEATURE_EXTRACTOR,
                                  resize_shape=RESIZE,
                                  n_channels=N_CHANNELS,
                                  autoencoder_units=AUTOENCODER_UNITS,
                                  initial_n_frames=MAX_FRAMES,
                                  units_mask=AUTOENCODER_MASK,
                                  finetune_feature_extractor=False
                                  )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f'{log_path}/ULSTM{MAX_FRAMES}_{"-".join(str(x) for x in zip(AUTOENCODER_UNITS, AUTOENCODER_MASK))}',
            monitor='val_loss',
            save_best_only=True, save_weights_only=True, save_format='tf'),
        keras.callbacks.TensorBoard(log_dir=f"{log_path}/{FEATURE_EXTRACTOR}/ULSTM{MAX_FRAMES}",
                                    write_graph=True, write_images=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto'),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.LearningRateScheduler(lr_scheduler),
#        keras.callbacks.BackupAndRestore(
#            backup_dir=f'BackupAndRestore/{log_path}/ULSTM{MAX_FRAMES}_{"-".join(str(x) for x in zip(AUTOENCODER_UNITS, AUTOENCODER_MASK))}'),
    ]
    model_instance.compile(optimizer='adam', loss='mse',
                           metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.CosineSimilarity(),
                                    tf.keras.metrics.LogCoshError()],
                           run_eagerly=True)
    model_instance.fit(x=train_generator,
                       validation_data=train_generator,
                       epochs=EPOCHS,
                       callbacks=callbacks)