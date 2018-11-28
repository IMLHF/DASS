import tensorflow as tf
import numpy as np
import librosa
import os
import shutil
import time
from . import mixed_aishell
import multiprocessing
import FLAGS


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parse_func(example_proto):
  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[FLAGS.input_size],
                                           dtype=tf.float32),
      'labels1': tf.FixedLenSequenceFeature(shape=[FLAGS.output_size],
                                            dtype=tf.float32),
      'labels2': tf.FixedLenSequenceFeature(shape=[FLAGS.output_size],
                                            dtype=tf.float32), }
  _, sequence = tf.parse_single_sequence_example(
      example_proto, sequence_features=sequence_features)
  length = tf.shape(sequence['inputs'])[0]
  return sequence['inputs'], sequence['labels1'], sequence['labels2'], length


def gen_single_tfrecord(i, index_name, dataset, dataset_dir):
  uttname1, uttname2 = str(index_name).split(' ')
  uttname1 = uttname1[uttname1.rfind('/')+1:uttname1.rfind('.')]
  uttname2 = uttname2[uttname2.rfind('/')+1:uttname2.rfind('.')]
  index_name = 'MIX'.join([uttname1, uttname2])
  X_Y = dataset.X_Y[i]
  X = np.reshape(np.array(X_Y[0], dtype=np.float32),
                 newshape=[-1, FLAGS.input_size])
  Y = np.reshape(np.array(X_Y[1], dtype=np.float32),
                 newshape=[-1, FLAGS.output_size*2])
  Y1 = Y[:, :FLAGS.output_size]
  Y2 = Y[:, FLAGS.output_size:]
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
      dataset = data_mixed.train
      if dataset_dir[-2:] == 'in':
        dataset = data_mixed.train
      elif dataset_dir[-2:] == 'on':
        dataset = data_mixed.validation
      elif dataset_dir[-2:] == 'cc':
        dataset = data_mixed.test_cc

      # pool = multiprocessing.Pool(FLAGS.num_threads_processing_data)
      # workers = []
      for i, index_name in enumerate(dataset.index_list):
        # workers.append(pool.apply_async(gen_single_tfrecord,
                                        # (i, index_name, dataset, dataset_dir)))
        gen_single_tfrecord(i, index_name, dataset, dataset_dir)
      # pool.close()
      # pool.join()
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
