from typing import Callable, Tuple, List, Union

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.layers.rnn.base_conv_lstm import ConvLSTMCell
from tensorflow.python.training.tracking.data_structures import NoDependency

from src.utils import crop_around_center_point, calculate_center_of_mass


class ULSTMModel(tf.keras.Model):

    def __init__(self, filters_kernel_size: List[Tuple[int, Tuple[int, int]]]):
        super(ULSTMModel, self).__init__()
        len_cells = len(filters_kernel_size)
        self.encoder_cells = [
            tf.keras.layers.ConvLSTM2D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=(1, 1),
                                       padding='same',
                                       return_sequences=True,
                                       return_state=True,
                                       name=f"encoder_cell_{idx}")
            for idx, (filters, kernel_size) in enumerate(filters_kernel_size)]
        self.decoder_cells = [
            tf.keras.layers.ConvLSTM2D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=(1, 1),
                                       padding='same',
                                       return_sequences=True if idx < len_cells - 1 else False,
                                       return_state=False,
                                       name=f"decoder_cell_{idx}")
            for idx, (filters, kernel_size) in enumerate(reversed(filters_kernel_size))]
        self.batch_normalization = [
            tf.keras.layers.BatchNormalization(name=f"batch_normalization_{idx}") for idx in range(len_cells)]

        self.downsample_layers = [tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) for idx in range(len_cells)]

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.ConvLSTM2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                       return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        ])
        self.conv1x1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                              name="Conv1x1")

    def call(self, inputs, training=False, mask=None):
        x = inputs
        states = []
        for idx, (encoder_cell, downsample_layer) in enumerate(zip(self.encoder_cells, self.downsample_layers)):
            x, c_t, h_t = encoder_cell(x, training=training)
            x = self.batch_normalization[idx](x)
            if idx < len(self.encoder_cells) - 1:
                x = downsample_layer(x)
            states.append((c_t, h_t))
        for idx, (decoder_cell, state) in enumerate(zip(self.decoder_cells, reversed(states))):
            x = decoder_cell(x, initial_state=state, training=training)
            if idx < len(self.decoder_cells) - 1:
                x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.UpSampling2D(size=(2, 2), name=f"Upsample2x_{idx}"))(x)
        x = self.conv1x1(x)
        x = x * inputs
        x = self.classifier(x)
        x = tf.reduce_max(x, axis=1)
        return x


class AttentionSeekingModule(tf.keras.Model):

    def __init__(self, num_heads: int = None,
                 key_dim: int = None,
                 loss_alpha: float = 0.01,
                 input_shape=(None, None, None, 1)):
        super(AttentionSeekingModule, self).__init__()
        if num_heads and key_dim:
            self.attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                                      key_dim=key_dim,
                                                                      name="Attention_Layer")
        else:
            self.attention_layer = tf.keras.layers.Attention(use_scale=True, name="Attention_Layer")
        self.conv1x1 = tf.keras.layers.Conv2D(filters=1,
                                              kernel_size=(1, 1),
                                              strides=(1, 1),
                                              padding='same',
                                              activation=None,
                                              name="Conv1x1",
                                              kernel_regularizer=tf.keras.regularizers.l2(loss_alpha))
        self.loss_alpha = loss_alpha
        self.coords = tf.convert_to_tensor(input_shape[1:3], dtype=tf.float32)

    def custom_loss(self, inputs):
        max_x = tf.reduce_max(inputs)
        return self.loss_alpha * tf.reduce_sum(1. / (max_x - inputs + 1e-10))

    def call(self, inputs, training=False, mask=None):
        x = inputs
        attention_out = self.attention_layer([x, x, x], training=training)
        x = self.conv1x1(attention_out)
        x = tf.sigmoid(x)
        # TODO: Add trainstep function with custom loss for each timestamp
        center_of_mass = calculate_center_of_mass(x)
        center_of_mass = (center_of_mass - self.coords / 2) / self.coords
        return attention_out, center_of_mass


class AttentionSeekingCell(tf.keras.Model):

    def __init__(self,
                 crop_size: tf.Tensor,
                 backbone_preprocessing_fn: Callable,
                 backbone_model: tf.keras.Model,
                 lstm_units: int,
                 num_heads: Union[int, None],
                 key_dim: Union[int, None],
                 pooling_layer: tf.keras.layers.Layer,
                 dropout_rate: float,
                 attention_seeking_module_alpha: float = 0.01):
        super(AttentionSeekingCell, self).__init__()
        self.crop_size = tf.cast(crop_size, tf.float32)
        self.backbone_preprocessing_fn = backbone_preprocessing_fn
        self.backbone_model = backbone_model(include_top=False,
                                             weights=None,
                                             input_shape=(*crop_size, 3),
                                             )
        self.backbone_model._name = "backbone_model"
        self.backbone_model.build((*crop_size, 3))
        self.lstm_cell = tf.keras.layers.LSTMCell(units=lstm_units)
        self.attention_seeking_module = AttentionSeekingModule(num_heads,
                                                               key_dim,
                                                               attention_seeking_module_alpha,
                                                               input_shape=self.backbone_model.output_shape)
        self.pooling_layer = pooling_layer
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.state_size = (self.lstm_cell.state_size, 2)
        self.new_coords = tf.Variable(tf.zeros(shape=(2,), dtype=tf.float32), trainable=False)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (self.lstm_cell.get_initial_state(inputs, batch_size, dtype),
                tf.random.uniform((2,), dtype=tf.float32))

    def build(self, input_shape):
        self.backbone_model.build(input_shape)
        output_shape = self.backbone_model.output_shape
        self.attention_seeking_module.build(output_shape)
        self.new_coords.assign(tf.random.uniform((2,), dtype=tf.float32))
        self.built = True

    def call(self, inputs, states=None, training=False):
        lstm_state, coords = states
        w_h = tf.cast(tf.shape(inputs)[1:3], tf.float32)
        image = crop_around_center_point(inputs, coords, self.crop_size)
        x = self.backbone_preprocessing_fn(image)
        x = self.backbone_model(x, training=training)
        # TODO: Run with updates
        x, coords_update = self.attention_seeking_module(x, training=training)
        self.new_coords.assign(coords + coords_update[0] * self.crop_size / w_h)
        if self.new_coords[0] < 0 or self.new_coords[0] > 1 or \
                self.new_coords[1] < 0 or self.new_coords[1] > 1:
            self.new_coords.assign(tf.random.uniform((2,), dtype=tf.float32))

        x = self.pooling_layer(x)
        x = self.dropout_layer(x, training=training)
        x, new_states = self.lstm_cell(x, lstm_state)
        # TODO: Add metric for new_coords (heatmap)
        return x, (new_states, self.new_coords)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, flattened=True, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        if flattened:
            # input is of shape (batch_size, frames, num_features)
            self.embeding_postprocess = tf.keras.layers.Lambda(lambda x: x)
        else:
            # input is of shape (batch_size, frames, rows, cols, num_features)
            self.embeding_postprocess = tf.keras.layers.Lambda(tf.reshape,
                                                               arguments={"shape": [-1,
                                                                                    self.sequence_length,
                                                                                    1,
                                                                                    1,
                                                                                    self.output_dim]})

    def call(self, inputs):
        """
        :param inputs: (batch_size, frames, num_features) if flattened,
        else (batch_size, frames, rows, cols, num_features)
        :return:
        """
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        embedded_positions = self.embeding_postprocess(embedded_positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads, attention_dropout, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=attention_dropout
        )
        self.dense_proj = tf.keras.Sequential(
            [tf.keras.layers.Dense(dense_dim, activation=tf.nn.gelu), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, mask=None, training=False):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask, training=training)
        proj_input = self.layernorm_1(inputs + attention_output, training=training)
        proj_output = self.dense_proj(proj_input, training=training)
        return self.layernorm_2(proj_input + proj_output)


class PyramidLevel(tf.keras.Model):

    def __init__(self,
                 sequence_length,
                 embed_dim,
                 dense_dim,
                 num_heads,
                 attention_dropout,
                 backbone_model,
                 flattened=True,
                 **kwargs):
        super(PyramidLevel, self).__init__(**kwargs)
        self.backbone_model = tf.keras.layers.TimeDistributed(backbone_model)
        self.embedding_layer = PositionalEmbedding(sequence_length,
                                                   embed_dim,
                                                   flattened=flattened,
                                                   name="frame_position_embedding")
        self.transformer_encoder = TransformerEncoder(embed_dim,
                                                      dense_dim,
                                                      num_heads,
                                                      attention_dropout,
                                                      name="transformer_encoder")

    def call(self, inputs, mask=None, training=False):
        """

        :param self:
        :param inputs: (batch_size, frames, width, heigh, channels)
        :param mask:
        :param training:
        :return:
        """
        x = self.backbone_model(inputs, training=training)
        x = self.embedding_layer(x, training=training)
        x = self.transformer_encoder(x, mask=mask, training=training)
        return x


class PyramidTransformer(tf.keras.Model):

    def __init__(self,
                 sequence_length: int,
                 dense_dim: Union[int, List[int]],
                 num_heads: Union[int, List[int]],
                 attention_dropout: Union[float, List[float]],
                 backbone_model: tf.keras.Model,
                 input_shape: Tuple[int, int, int],
                 backbone_pooling: Union[str, None] = None,
                 final_pooling: Union[str, None] = None,
                 n_pyramid_levels: Union[int, None] = 3,
                 **kwargs
                 ):
        """

        :param sequence_length: Number of frames
        :param dense_dim:
        :param num_heads:
        :param attention_dropout:
        :param backbone_preprocess:
        :param backbone_model:
        :param input_shape:
        :param backbone_pooling:
        :param n_pyramid_levels:
        :param kwargs:
        """
        super(PyramidTransformer, self).__init__(**kwargs)
        self.trainable_levels = n_pyramid_levels
        if not backbone_pooling:
            self.upsample = True
        else:
            self.upsample = False
        self.flattened = not self.upsample
        self.input_layer = tf.keras.layers.Input(shape=(sequence_length, *input_shape))
        w, h, c = input_shape
        self.pyramid_levels = []
        for lvl_idx in range(n_pyramid_levels):
            if isinstance(dense_dim, int):
                dense_dim_lvl = dense_dim
            else:
                dense_dim_lvl = dense_dim[lvl_idx]
            if isinstance(num_heads, int):
                num_heads_lvl = num_heads
            else:
                num_heads_lvl = num_heads[lvl_idx]
            if isinstance(attention_dropout, float):
                attention_dropout_lvl = attention_dropout
            else:
                attention_dropout_lvl = attention_dropout[lvl_idx]
            backbone_lvl = backbone_model(include_top=False,
                                          weights=None,
                                          input_shape=(w, h, 3),
                                          pooling=backbone_pooling)
            backbone_lvl.build(input_shape=(w, h, 3))
            embed_dim_lvl = backbone_lvl.output_shape[-1]
            if self.upsample and lvl_idx != 0:  # top-most level is not upsampled
                upsample_lvl = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(2, 2)),
                                                               name="upsample_lvl_{}".format(lvl_idx))
            else:
                upsample_lvl = tf.keras.layers.Lambda(lambda x: x,
                                                      name="upsample_lvl_{}".format(lvl_idx))
            self.pyramid_levels.append(
                (upsample_lvl,
                 self.create_pyramid_level(self.input_layer,
                                           sequence_length,
                                           embed_dim_lvl,
                                           dense_dim_lvl,
                                           num_heads_lvl,
                                           attention_dropout_lvl,
                                           backbone_lvl,
                                           flattened=self.flattened,
                                           input_shape=(w, h, 3),
                                           name=f'pyramid_level_{lvl_idx}'))
            )
            w = w // 2
            h = h // 2
        self.initial_output_shape = self.pyramid_levels[-1][1].output_shape
        if final_pooling:
            self.final_pooling = tf.keras.layers.TimeDistributed(final_pooling)
        else:
            self.final_pooling = tf.keras.layers.Lambda(lambda x: x)
        self.classifier = tf.keras.Sequential(
            [tf.keras.layers.GlobalAveragePooling1D(),
             tf.keras.layers.Dropout(0.5),
             tf.keras.layers.Dense(1, activation='sigmoid')]
        )

    @staticmethod
    def create_pyramid_level(inputs: tf.keras.layers.Input,
                             sequence_length: int,
                             embed_dim: int,
                             dense_dim: int,
                             num_heads: int,
                             attention_dropout: float,
                             backbone_model: tf.keras.Model,
                             flattened: bool,
                             input_shape: Tuple[int, int, int],
                             name: str = None,
                             **kwargs):

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Resizing(input_shape[0], input_shape[1]))(inputs)
        lvl_model = PyramidLevel(sequence_length,
                                 embed_dim,
                                 dense_dim,
                                 num_heads,
                                 attention_dropout,
                                 backbone_model,
                                 flattened,
                                 name=name,
                                 **kwargs)
        outputs = lvl_model(x)
        return tf.keras.Model(inputs, outputs, name=name)

    def set_trainable_levels(self, trainable_levels: int):
        self.trainable_levels = trainable_levels

    def call(self, inputs, training=None, mask=None):
        ret = tf.zeros((tf.shape(inputs)[0], *self.initial_output_shape[1:]), dtype=tf.float32)
        for upsample, model in list(reversed(self.pyramid_levels))[:self.trainable_levels]:
            x = model(inputs, training=training)
            ret = ret + x
            ret = upsample(ret, training=training)
        # upsample to keep the dimension
        for upsample, _ in list(reversed(self.pyramid_levels))[self.trainable_levels:]:
            ret = upsample(ret, training=training)
        ret = self.final_pooling(ret, training=training)
        ret = self.classifier(ret, training=training)
        return ret
