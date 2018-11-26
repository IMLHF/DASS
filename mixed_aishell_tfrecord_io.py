import tensorflow as tf
import numpy as np
import librosa
import os
import shutil

LEN_WAVE_PADDING_TO = 160000
SR = 16000
SPEAKER_NUM = 4
TRAIN_SET_UTT_NUM_PER_SPEAKER = 260
VAL_SET_UTT_NUM_PER_SPEAKER = 30


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def load_wave_mat(speakerlist_dir):
  speaker_list = list(os.listdir(speakerlist_dir))
  speaker_mat = []
  for speaker_name in speaker_list:
    utt_mat = []
    speaker_dir = os.path.join(speakerlist_dir, speaker_name)
    if not os.path.isdir(speaker_dir):
      continue
    utt_list = list(os.listdir(speaker_dir))
    for utt_name in utt_list[:TRAIN_SET_UTT_NUM_PER_SPEAKER+VAL_SET_UTT_NUM_PER_SPEAKER]:
      utt_dir = os.path.join(speaker_dir, utt_name)
      waveData, sr = librosa.load(utt_dir, sr=SR)
      # waveData=(waveData / np.max(np.abs(waveData)))* 32767 # 数据在（-1,1）之间，在读取后变换为16bit，节省空间
      while len(waveData) < LEN_WAVE_PADDING_TO:
        waveData = np.tile(waveData, 2)
      utt_mat.append(waveData[:LEN_WAVE_PADDING_TO])
    speaker_mat.append(np.array(utt_mat, dtype=np.float32))
    # print(np.shape(np.array(utt_mat,dtype=np.float32)))
    print(speaker_name+' over')
  speaker_mat = np.array(speaker_mat, dtype=np.float32)
  print(np.shape(speaker_mat))
  train, val, _ = np.split(speaker_mat,
                           [TRAIN_SET_UTT_NUM_PER_SPEAKER,
                            TRAIN_SET_UTT_NUM_PER_SPEAKER+VAL_SET_UTT_NUM_PER_SPEAKER],
                           axis=1)
  return train, val


def gen_tfrecord(data_dir, gen=True):
  if data_dir[-1] == '/':
    data_dir = data_dir[:-1]
  speakerlist_dir = data_dir

  if not gen:
    return speakerlist_dir+'_train.tfrecords', speakerlist_dir+'_val.tfrecords'

  speech_train, speech_val = load_wave_mat(speakerlist_dir)
  speech_train_shape = np.shape(speech_train)
  speech_train = np.reshape(speech_train, [-1])
  speech_val_shape = np.shape(speech_val)
  speech_val = np.reshape(speech_val, [-1])

  # 1. writer
  writer = tf.python_io.TFRecordWriter(speakerlist_dir+'_train.tfrecords')

  # 2. example
  record = tf.train.Example(
      features=tf.train.Features(
          feature={
              # 'mat': _bytes_feature(speech.tostring())
              'shape': _int64_feature(speech_train_shape),
              'mat': _float_feature(speech_train)
          }))
  # serialize and write
  writer.write(record.SerializeToString())
  writer.close()

  writer = tf.python_io.TFRecordWriter(speakerlist_dir+'_val.tfrecords')
  record = tf.train.Example(
      features=tf.train.Features(
          feature={
              # 'mat': _bytes_feature(speech.tostring())
              'shape': _int64_feature(speech_val_shape),
              'mat': _float_feature(speech_val)
          }))
  writer.write(record.SerializeToString())
  writer.close()
  return speakerlist_dir+'_train.tfrecords', speakerlist_dir+'_val.tfrecords'


def tf_normalize_spec(spec):
  upbound = 7.0
  lowbound = -1.0
  spec = tf.divide(tf.log(spec+0.5), tf.log(10.0))
  spec = tf.clip_by_value(spec, lowbound, upbound)
  spec = tf.subtract(spec, lowbound)
  spec = tf.divide(spec, tf.subtract(upbound, lowbound))
  return spec


def tf_rmNormalization_spec(spec):
  upbound = 7.0
  lowbound = -1.0
  spec = tf.math.pow(10.0, spec)-0.5
  spec = tf.where(spec > 0.0, spec, 0.0)
  spec = tf.multiply(spec*(upbound-lowbound))+lowbound
  return spec


def extract_Feature(example_proto, features):
  dataset = tf.parse_single_example(example_proto, features)
  # shape=dataset['shape']
  mat = tf.cast(dataset['mat'], tf.float32)
  speaker_num = tf.shape(mat)[0]
  speakerid = tf.random_uniform([2], maxval=speaker_num, dtype=tf.int32)
  speakerid = tf.while_loop(lambda sid: tf.equal(sid[0], sid[1]),
                            lambda sid: (tf.random_uniform([2],
                                                           maxval=speaker_num,
                                                           dtype=tf.int32),),
                            (speakerid,))
  speaker1id = speakerid[0]
  speaker2id = speakerid[1]
  utt1id = tf.random_uniform([], maxval=tf.shape(mat)[1], dtype=tf.int32)
  utt2id = tf.random_uniform([], maxval=tf.shape(mat)[1], dtype=tf.int32)
  waveData1 = mat[speaker1id][utt1id]
  waveData2 = mat[speaker2id][utt2id]
  waveData1 = tf.multiply(
      tf.divide(waveData1, tf.reduce_max(tf.abs(waveData1))), 32767.0)
  waveData2 = tf.multiply(
      tf.divide(waveData2, tf.reduce_max(tf.abs(waveData2))), 32767.0)

  mixed_wave = tf.divide(tf.cast(tf.add(waveData1, waveData2), tf.float32), 2)
  mixed_spec = tf.abs(tf.contrib.signal.stft(signals=mixed_wave,
                                             frame_length=512,
                                             frame_step=256,
                                             fft_length=512))
  wave1_spec = tf.abs(tf.contrib.signal.stft(signals=waveData1,
                                             frame_length=512,
                                             frame_step=256,
                                             fft_length=512))
  wave2_spec = tf.abs(tf.contrib.signal.stft(signals=waveData2,
                                             frame_length=512,
                                             frame_step=256,
                                             fft_length=512))
  mixed_spec = tf_normalize_spec(mixed_spec)
  wave1_spec = tf_normalize_spec(wave1_spec)
  wave2_spec = tf_normalize_spec(wave2_spec)
  return mixed_spec, wave1_spec, wave2_spec, tf.shape(wave1_spec)[0]


def extract_X_Y_Lengths_train(example_proto):
  # return example_proto,tf.random_uniform(shape=[]),233,233
  features = {
      'mat': tf.FixedLenFeature([SPEAKER_NUM, TRAIN_SET_UTT_NUM_PER_SPEAKER, LEN_WAVE_PADDING_TO],
                                tf.float32),
      'shape': tf.FixedLenFeature([3], tf.int64),
  }
  return extract_Feature(example_proto, features)


def extract_X_Y_Lengths_val(example_proto):
  # return example_proto,tf.random_uniform(shape=[]),233,233
  features = {
      'mat': tf.FixedLenFeature([SPEAKER_NUM, VAL_SET_UTT_NUM_PER_SPEAKER, LEN_WAVE_PADDING_TO],
                                tf.float32),
      'shape': tf.FixedLenFeature([3], tf.int64),
  }
  return extract_Feature(example_proto, features)


if __name__ == "__main__":
  gen_tfrecord()
