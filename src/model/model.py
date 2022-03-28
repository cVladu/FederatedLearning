import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_feature_extractor(model='InceptionV3', input_shape=None):
    extractor_dict = \
        {
            'InceptionV3': keras.applications.inception_v3.InceptionV3,
            'VGG16': keras.applications.vgg16.VGG16,
            'VGG19': keras.applications.vgg19.VGG19,
            'ResNet50': keras.applications.resnet.ResNet50,
            'Xception': keras.applications.xception.Xception,
            'InceptionResNetV2': keras.applications.inception_resnet_v2.InceptionResNetV2,
            'MobileNet': keras.applications.mobilenet.MobileNet,
            'MobileNetV2': keras.applications.mobilenet_v2.MobileNetV2,
            'DenseNet121': keras.applications.densenet.DenseNet121,
            'DenseNet169': keras.applications.densenet.DenseNet169,
            'DenseNet201': keras.applications.densenet.DenseNet201,
            'EfficientNetB0': keras.applications.efficientnet.EfficientNetB0,
            'EfficientNetB1': keras.applications.efficientnet.EfficientNetB1,
            'EfficientNetB2': keras.applications.efficientnet.EfficientNetB2,
            'EfficientNetB3': keras.applications.efficientnet.EfficientNetB3,
            'EfficientNetB4': keras.applications.efficientnet.EfficientNetB4,
            'EfficientNetB5': keras.applications.efficientnet.EfficientNetB5,
            'EfficientNetB6': keras.applications.efficientnet.EfficientNetB6,
            'EfficientNetB7': keras.applications.efficientnet.EfficientNetB7,
        }

    try:
        feature_extractor = extractor_dict[model](include_top=False, weights='imagenet',
                                                  input_shape=input_shape, pooling='avg')
    except KeyError:
        print("Model not found")
        print("Falling back to InceptionV3")
        feature_extractor = extractor_dict['InceptionV3'](include_top=False, weights='imagenet',
                                                          input_shape=input_shape, pooling='avg')
    preprocess_dict = \
        {
            'InceptionV3': keras.applications.inception_v3.preprocess_input,
            'VGG16': keras.applications.vgg16.preprocess_input,
            'VGG19': keras.applications.vgg19.preprocess_input,
            'ResNet50': keras.applications.resnet.preprocess_input,
            'Xception': keras.applications.xception.preprocess_input,
            'InceptionResNetV2': keras.applications.inception_resnet_v2.preprocess_input,
            'MobileNet': keras.applications.mobilenet.preprocess_input,
            'MobileNetV2': keras.applications.mobilenet_v2.preprocess_input,
            'DenseNet121': keras.applications.densenet.preprocess_input,
            'DenseNet169': keras.applications.densenet.preprocess_input,
            'DenseNet201': keras.applications.densenet.preprocess_input,
            'NASNetLarge': keras.applications.nasnet.preprocess_input,
            'NASNetMobile': keras.applications.nasnet.preprocess_input,
            'EfficientNetB0': keras.applications.efficientnet.preprocess_input,
            'EfficientNetB1': keras.applications.efficientnet.preprocess_input,
            'EfficientNetB2': keras.applications.efficientnet.preprocess_input,
            'EfficientNetB3': keras.applications.efficientnet.preprocess_input,
            'EfficientNetB4': keras.applications.efficientnet.preprocess_input,
            'EfficientNetB5': keras.applications.efficientnet.preprocess_input,
            'EfficientNetB6': keras.applications.efficientnet.preprocess_input,
            'EfficientNetB7': keras.applications.efficientnet.preprocess_input,
        }
    try:
        preprocess = preprocess_dict[model]
    except KeyError:
        print("Model not found")
        print("Falling back to InceptionV3")
        preprocess = preprocess_dict['InceptionV3']

    inputs = keras.Input(shape=input_shape)
    preprocessed = preprocess(inputs)
    features = feature_extractor(preprocessed)
    return keras.Model(inputs, features)


def build_autoencoder(feature_extractor: tf.keras.Model = None,
                      units=None,
                      units_mask=None,
                      initial_n_frames: int = 0) -> tf.keras.Model:

    if initial_n_frames <= 0:
        raise ValueError("initial_n_frames must be greater than 0")

    if units is None or units == []:
        units = [128]

    # Inputs
    inputs = tf.keras.layers.Input(shape=(initial_n_frames, feature_extractor.output_shape[1]))

    if len(units) == 1:
        # Only one encoder and decoder
        encoder = tf.keras.layers.LSTM(units=units[0], return_state=True)
        encoder_outputs, state_h, state_c = encoder(inputs)

        decoder_lstm = tf.keras.layers.LSTM(units=units[0], return_state=True, return_sequences=False)
        decoder_outputs, _, _ = decoder_lstm(inputs, initial_state=[state_h, state_c])
    else:
        internal_states = []
        # Encoder

        # First encoder
        encoder = tf.keras.layers.LSTM(units=units[0], return_state=True, return_sequences=True, name='encoder_0')
        encoder_outputs, state_h, state_c = encoder(inputs)
        internal_states.append([state_h, state_c])

        # Residual encoder and decoder
        for i, unit in enumerate(units[1:-1]):
            encoder = tf.keras.layers.LSTM(units=unit, return_state=True, return_sequences=True, name=f'encoder_{i+1}')
            encoder_outputs, state_h, state_c = encoder(encoder_outputs)
            internal_states.append([state_h, state_c])

        # Last encoder
        encoder = tf.keras.layers.LSTM(units=units[-1],
                                       return_state=True,
                                       return_sequences=False,
                                       name=f'encoder_{len(units)-1}')
        encoder_outputs, state_h, state_c = encoder(encoder_outputs)
        internal_states.append([state_h, state_c])

        # Decoder

        # First decoder
        decoder = tf.keras.layers.LSTM(units=units[-1], return_state=True, return_sequences=True, name='decoder_0')
        if units_mask[0]:
            decoder_outputs, _, _ = decoder(inputs, initial_state=internal_states[-1])
        else:
            decoder_outputs, _, _ = decoder(inputs)

        for i, (unit, internal_state, mask) in enumerate(zip(units[0:-1:-1], internal_states[0:-1:-1], units_mask)):
            decoder = tf.keras.layers.LSTM(units=unit, return_sequences=True, name=f'decoder_{i+1}')
            if mask:
                decoder_outputs = decoder(decoder_outputs, initial_state=internal_state)
            else:
                decoder_outputs = decoder(decoder_outputs)

        # Last decoder
        decoder = tf.keras.layers.LSTM(units=units[0],
                                       return_state=True,
                                       return_sequences=False,
                                       name=f'decoder_{len(units)-1}')
        if units_mask[-1]:
            decoder_outputs, _, _ = decoder(decoder_outputs, initial_state=internal_states[0])
        else:
            decoder_outputs, _, _ = decoder(decoder_outputs)

    decoder_dense = tf.keras.layers.Dense(units=feature_extractor.output_shape[1], activation='relu')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = tf.keras.Model(inputs=inputs, outputs=decoder_outputs)
    return model


class ULSTMModel(tf.keras.Model):

    def __init__(self, feature_extractor: str,
                 resize_shape,
                 n_channels,
                 autoencoder_units,
                 initial_n_frames: int = 0,
                 units_mask=None,
                 finetune_feature_extractor: bool = False,
                 **kwargs):
        super(ULSTMModel, self).__init__(**kwargs)
        self.feature_extractor = build_feature_extractor(feature_extractor, input_shape=(*resize_shape, n_channels))
        self.feature_extractor.trainable = finetune_feature_extractor
        self.autoencoder = build_autoencoder(feature_extractor=self.feature_extractor,
                                             units=autoencoder_units,
                                             units_mask=units_mask,
                                             initial_n_frames=initial_n_frames)

    def __call__(self, inputs, training=True):
        x = np.empty(shape=(inputs.shape[0], inputs.shape[1], self.feature_extractor.output_shape[1]))
        for frame_idx in range(inputs.shape[1]):
            x[:, frame_idx, ...] = self.feature_extractor(inputs[:, frame_idx, ...])
        return self.autoencoder(x, training=training)
