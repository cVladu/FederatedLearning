import numpy as np
import tensorflow as tf


@tf.function(jit_compile=True,
             input_signature=(tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32), ))
def calculate_center_of_mass(img_tensor):
    ii, jj = tf.meshgrid(tf.range(tf.shape(img_tensor)[-3]), tf.range(tf.shape(img_tensor)[-2]), indexing='ij')
    coords = tf.stack([tf.reshape(ii, (-1,)), tf.reshape(jj, (-1,))], axis=-1)
    img_flat = tf.reshape(img_tensor, [-1, tf.shape(img_tensor)[1] * tf.shape(img_tensor)[2], 1])
    total_mass = tf.reduce_sum(img_flat)
    center_of_mass = tf.reduce_sum(tf.cast(coords, tf.float32) * img_flat, axis=1) / total_mass
    return center_of_mass


@tf.function(jit_compile=True,
             input_signature=(tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.float32),
                              tf.TensorSpec(shape=[2, ], dtype=tf.float32),
                              tf.TensorSpec(shape=[2, ], dtype=tf.float32)))
def crop_around_center_point(image: tf.Tensor, center_point: tf.Tensor, crop_size: tf.Tensor):
    """
    Crop the image around the center point.
    :param image: image to crop. Shape: [None, height, width, channels]
    :param center_point: center point of the image
    :param crop_size: size of the crop
    :return: cropped image
    """
    h = tf.cast(tf.shape(image)[-3], tf.float32)
    w = tf.cast(tf.shape(image)[-2], tf.float32)
    # if crop_size[0] > h or crop_size[1] > w:
    #     return tf.cast(tf.image.resize(image, tf.cast(crop_size, tf.int32)), tf.float16)
    # Calculate the top left corner of the crop
    top_left_corner = center_point - 1. / crop_size / 2
    offset_h = h * top_left_corner[0]
    offset_w = w * top_left_corner[1]
    offset_h = tf.clip_by_value(offset_h, 0, h - crop_size[0])
    offset_h = tf.cast(offset_h, tf.int32)
    offset_w = tf.clip_by_value(offset_w, 0, w - crop_size[1])
    offset_w = tf.cast(offset_w, tf.int32)
    crop_size = tf.cast(crop_size, tf.int32)
    return tf.image.crop_to_bounding_box(image,
                                         offset_h,
                                         offset_w,
                                         crop_size[0],
                                         crop_size[1])


if __name__ == "__main__":
    test_img = np.zeros((1, 16, 16, 1), dtype="float32")
    test_img[-1, -1, -1, -1] = 1
    ret = calculate_center_of_mass(tf.convert_to_tensor(test_img))
    pass