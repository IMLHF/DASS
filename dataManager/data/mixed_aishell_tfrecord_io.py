import tensorflow as tf
import numpy as np
import librosa
import os
import shutil
import time
from . import mixed_aishell
import multiprocessing
import copy
import scipy.io
from FLAGS import NNET_PARAM


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parse_func(example_proto):
  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.input_size],
                                           dtype=tf.float32),
      'labels1': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.output_size],
                                            dtype=tf.float32),
      'labels2': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.output_size],
                                            dtype=tf.float32), }
  _, sequence = tf.parse_single_sequence_example(
      example_proto, sequence_features=sequence_features)
  length = tf.shape(sequence['inputs'])[0]
  return sequence['inputs'], sequence['labels1'], sequence['labels2'], length


def gen_tfrecord_minprocess(dataset,s_site,e_site, dataset_dir):
  # for (i, index_) in enumerate(dataset_index_list):
  for i in range(s_site,e_site):
    X_Y=dataset.X_Y[i]
    # X_Y = mixed_aishell._extract_feature_x_y([index_, ])
    X = np.reshape(np.array(X_Y[0], dtype=np.float32),
                   newshape=[-1, NNET_PARAM.input_size])
    Y = np.reshape(np.array(X_Y[1], dtype=np.float32),
                   newshape=[-1, NNET_PARAM.output_size*2])
    Y1 = Y[:, :NNET_PARAM.output_size]
    Y2 = Y[:, NNET_PARAM.output_size:]
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in X]
    label_features1 = [
        tf.train.Feature(float_list=tf.train.FloatList(value=label))
        for label in Y1]
    label_features2 = [
        tf.train.Feature(float_list=tf.train.FloatList(value=label))
        for label in Y2]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels1': tf.train.FeatureList(feature=label_features1),
        'labels2': tf.train.FeatureList(feature=label_features2),
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    record = tf.train.SequenceExample(feature_lists=feature_lists)
    writer = tf.python_io.TFRecordWriter(os.path.join(
        dataset_dir, ('%08d.tfrecords' % i)))
    writer.write(record.SerializeToString())
    writer.close()
    # print(dataset_dir + ('/%08d.tfrecords' % i), 'write done')


def gen_tfrecord(data_dir, tfrecords_dir, gen=True):
  train_tfrecords_dir = os.path.join(tfrecords_dir, 'train')
  val_tfrecords_dir = os.path.join(tfrecords_dir, 'validation')
  testcc_tfrecords_dir = os.path.join(tfrecords_dir, 'test_cc')
  dataset_dir_list = [train_tfrecords_dir,
                      val_tfrecords_dir, testcc_tfrecords_dir]

  if gen:
    if os.path.exists(train_tfrecords_dir):
      shutil.rmtree(train_tfrecords_dir)
    if os.path.exists(val_tfrecords_dir):
      shutil.rmtree(val_tfrecords_dir)
    if os.path.exists(testcc_tfrecords_dir):
      shutil.rmtree(testcc_tfrecords_dir)
    os.makedirs(train_tfrecords_dir)
    os.makedirs(val_tfrecords_dir)
    os.makedirs(testcc_tfrecords_dir)

    data_mixed = mixed_aishell.read_data_sets(data_dir)
    gen_start_time = time.time()
    for dataset_dir in dataset_dir_list:
      start_time = time.time()
      # dataset_index_list = None
      dataset = None
      if dataset_dir[-2:] == 'in':
        dataset = data_mixed.train
        # dataset_index_list = scipy.io.loadmat(
        #     '_data/mixed_aishell/train/mixed_wav_dir.mat')["mixed_wav_dir"]
      elif dataset_dir[-2:] == 'on':
        dataset = data_mixed.validation
        # dataset_index_list = scipy.io.loadmat(
        #     '_data/mixed_aishell/validation/mixed_wav_dir.mat')["mixed_wav_dir"]
      elif dataset_dir[-2:] == 'cc':
        dataset = data_mixed.test_cc
        # dataset_index_list = scipy.io.loadmat(
        #     '_data/mixed_aishell/test_cc/mixed_wav_dir.mat')["mixed_wav_dir"]

      # len_dataset = len(dataset_index_list)
      len_dataset = len(dataset.index_list)
      minprocess_utt_num = int(
          len_dataset/NNET_PARAM.num_threads_processing_data)
      pool = multiprocessing.Pool(NNET_PARAM.num_threads_processing_data)
      for i_process in range(NNET_PARAM.num_threads_processing_data):
        s_site = i_process*minprocess_utt_num
        e_site = s_site+minprocess_utt_num
        if i_process == (NNET_PARAM.num_threads_processing_data-1):
          e_site = len_dataset
        # print(s_site,e_site)
        pool.apply_async(gen_tfrecord_minprocess,
                         (copy.deepcopy(dataset),
                          copy.deepcopy(s_site),
                          copy.deepcopy(e_site),
                          copy.deepcopy(dataset_dir)))
        # gen_tfrecord_minprocess(copy.deepcopy(dataset_index_list[s_site:e_site]),
        #                         copy.deepcopy(dataset_dir))
      pool.close()
      pool.join()

      print(dataset_dir+' set extraction over. cost time %06d' %
            (time.time()-start_time))
    print('Generate TFRecord over. cost time %06d' %
          (time.time()-gen_start_time))

  train_set = os.path.join(train_tfrecords_dir, '*.tfrecords')
  val_set = os.path.join(val_tfrecords_dir, '*.tfrecords')
  testcc_set = os.path.join(testcc_tfrecords_dir, '*.tfrecords')
  return train_set, val_set, testcc_set


if __name__ == "__main__":
  gen_tfrecord()
