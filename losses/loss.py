import tensorflow as tf
import numpy as np


def MSE(y1, y2):
  return tf.reduce_mean(tf.square(tf.subtract(y1, y2)))


def utt_PIT_MSE_for_CNN(y1, y2):
  # for i in range(tf.shape(y1)[0]):
  loss1=tf.reduce_mean(tf.square(tf.subtract(y1, y2)),[1,2])
  y1_speaker1, y1_speaker2 = tf.split(y1, 2, axis=-1)
  y1_swaped = tf.concat([y1_speaker2, y1_speaker1], axis=-1)
  loss2=tf.reduce_mean(tf.square(tf.subtract(y1_swaped, y2)),[1,2])
  loss=tf.where(tf.less(loss1,loss2),loss1,loss2)
  return tf.reduce_mean(loss)

def frame_PIT_MSE_for_CNN(y1, y2):
  # for i in range(tf.shape(y1)[0]):
  loss1=tf.reduce_mean(tf.square(tf.subtract(y1, y2)),axis=2)
  y1_speaker1, y1_speaker2 = tf.split(y1, 2, axis=-1)
  y1_swaped = tf.concat([y1_speaker2, y1_speaker1], axis=-1)
  loss2=tf.reduce_mean(tf.square(tf.subtract(y1_swaped, y2)),axis=2)
  loss=tf.where(tf.less(loss1,loss2),loss1,loss2)
  return tf.reduce_mean(loss)


def PIT_MSE_abandon(y1, y2):
  # np.shape(y1)
  axis = len(np.shape(y1))-1
  y1_speaker1, y1_speaker2 = tf.split(y1, 2, axis=axis)
  y1_swaped = tf.concat([y1_speaker2, y1_speaker1], axis=axis)
  return tf.minimum(tf.reduce_mean(tf.square(tf.subtract(y1, y2))),
                    tf.reduce_mean(tf.square(tf.subtract(y1_swaped, y2))))
