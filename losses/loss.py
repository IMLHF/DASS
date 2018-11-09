import tensorflow as tf
import numpy as np


def MSE(y1, y2):
  return tf.reduce_mean(tf.square(tf.subtract(y1, y2)))


def PIT_MSE(y1, y2):
  # np.shape(y1)
  axis = len(np.shape(y1))-1
  y1_speaker1, y1_speaker2 = tf.split(y1, 2, axis=axis)
  y1_swaped = tf.concat([y1_speaker2, y1_speaker1], axis=axis)
  return tf.minimum(tf.reduce_mean(tf.square(tf.subtract(y1, y2))),
                    tf.reduce_mean(tf.square(tf.subtract(y1_swaped, y2))))
