import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer


class attention_layer(Layer):
    def __init__(self, input_shape):
        super(attention_layer, self).__init__()
        self.W = self.add_weight(name='attention_layer_weight', shape=(input_shape))