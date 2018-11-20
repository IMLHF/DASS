#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017   Sining Sun (Northwestern Polytechnical University, China)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from models.lstm_pit import LSTM
from dataManager.data import mixed_aishell
import utils
import wave

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# from io_funcs.tfrecords_io import get_padded_batch
# from local.utils import pp, show_all_variables

FLAGS = None

def decode():
  """Decoding the inputs using current model."""

  # data_dir = '/home/student/work/pit_test/data_small'
  speech_num = 10
  data_dir = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set'
  data_mixed = mixed_aishell.read_data_sets(data_dir)

  with tf.Graph().as_default():
    with tf.name_scope('model'):
      model = LSTM(FLAGS, infer=True)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    sess = tf.Session()

    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir+'/nnet')
    if ckpt and ckpt.model_checkpoint_path:
      tf.logging.info("Restore from " + ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      tf.logging.fatal("checkpoint not found.")
      sys.exit(-1)

  data_dir = FLAGS.data_dir
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  try:
    test_cc_X_Y = data_mixed.test_cc.X_Y[:speech_num]
    angle_batch = data_mixed.test_cc.X_Theta[:speech_num]
    x_batch = test_cc_X_Y[0]
    y_batch = test_cc_X_Y[1]
    lengths = np.array([np.shape(x_batch)[1]]*np.shape(x_batch)[0])
    cleaned1, cleaned2 = sess.run(
        [model.cleaned1, model.cleaned2],
        feed_dict={
            model.inputs: x_batch,
            model.labels: y_batch,
            model.lengths: lengths,
        })

    raw_spec1, raw_spec2 = np.split(y_batch, 2, axis=-1)

    cleaned1 = mixed_aishell.rmNormalization(cleaned1)
    cleaned2 = mixed_aishell.rmNormalization(cleaned2)
    raw_spec1 = mixed_aishell.rmNormalization(raw_spec1)
    raw_spec2 = mixed_aishell.rmNormalization(raw_spec2)

    if FLAGS.decode_show_spec:
      cleaned=np.concatenate([cleaned1,cleaned2],axis=-1)
      raw_spec=np.concatenate([raw_spec1,raw_spec2],axis=-1)
      utils.spectrum_tool.picture_spec(cleaned,
                                       'exp/lstm_pit/decode_ans/restore_spec_')
      utils.spectrum_tool.picture_spec(raw_spec,
                                       'exp/lstm_pit/decode_ans/raw_spec_')

    spec1 = cleaned1 * np.exp(angle_batch*1j)
    spec2 = cleaned2 * np.exp(angle_batch*1j)
    raw_spec1 *= np.exp(angle_batch*1j)
    raw_spec2 *= np.exp(angle_batch*1j)

    for i in range(speech_num):
      reY1 = utils.spectrum_tool.librosa_istft(
          spec1[i].T, (FLAGS.input_size-1)*2, FLAGS.input_size-1)
      reY2 = utils.spectrum_tool.librosa_istft(
          spec2[i].T, (FLAGS.input_size-1)*2, FLAGS.input_size-1)
      tmpY = np.concatenate([reY1, reY2])
      wavefile = wave.open(
          'exp/lstm_pit/decode_ans/restore_audio_'+str(i)+'.wav', 'wb')
      nchannels = 1
      sampwidth = 2  # 采样位宽，2表示16位
      framerate = 16000
      nframes = len(tmpY)
      comptype = "NONE"
      compname = "not compressed"
      wavefile.setparams((nchannels, sampwidth, framerate, nframes,
                          comptype, compname))
      wavefile.writeframes(
          np.array(tmpY, dtype=np.int16))

      rawY1 = utils.spectrum_tool.librosa_istft(
          raw_spec1[i].T, (FLAGS.input_size-1)*2, FLAGS.input_size-1)
      rawY2 = utils.spectrum_tool.librosa_istft(
          raw_spec2[i].T, (FLAGS.input_size-1)*2, FLAGS.input_size-1)
      tmpY = np.concatenate([rawY1, rawY2])
      wavefile = wave.open(
          'exp/lstm_pit/restore_audio/raw_audio_'+str(i)+'.wav', 'wb')
      nframes = len(tmpY)
      wavefile.setparams((nchannels, sampwidth, framerate, nframes,
                          comptype, compname))
      wavefile.writeframes(
          np.array(tmpY, dtype=np.int16))

  except Exception as e:
    print(e)
  finally:
    pass
  tf.logging.info("Done decoding.")
  sess.close()
  ''''''


def train_one_epoch(sess, coord, tr_model, data):
  """Runs the model one epoch on given data."""
  tr_loss = 0
  x_len = len(data)
  total_batch = x_len//FLAGS.batch_size if (x_len % FLAGS.batch_size == 0) else ((
      x_len//FLAGS.batch_size)+1)
  for i in range(total_batch):
    s_site = i*FLAGS.batch_size
    e_site = min(s_site+FLAGS.batch_size, x_len)
    x_y = data[s_site:e_site]
    x_batch = np.array(x_y[0], dtype=np.float32)
    y_batch = np.array(x_y[1], dtype=np.float32)
    lengths = np.array([np.shape(x_batch)[1]]*np.shape(x_batch)[0])
    if coord.should_stop():
      break
    _, loss = sess.run([tr_model.train_op, tr_model.loss],
                       feed_dict={tr_model.inputs: x_batch,
                                  tr_model.labels: y_batch,
                                  tr_model.lengths: lengths})
    tr_loss += loss
    # print('train %d loss %lf' % (i+1, loss/(e_site-s_site)))
    if (i+1) % 100 == 0:
      lr = sess.run(tr_model.lr)
      print("MINIBATCH %d: TRAIN AVG.LOSS %f, "
            "(learning rate %e)" % (
                i + 1, tr_loss / (i*FLAGS.batch_size+e_site-s_site), lr))
      sys.stdout.flush()
  tr_loss /= x_len
  return tr_loss


def eval_one_epoch(sess, coord, val_model, data):
  """Cross validate the model on given data."""
  val_loss = 0
  x_len = len(data)
  total_batch = x_len//FLAGS.batch_size if (x_len % FLAGS.batch_size == 0) else ((
      x_len//FLAGS.batch_size)+1)
  # print('val batch num %d' % total_batch)
  # print('len %d' % x_len)
  for i in range(total_batch):
    s_site = i*FLAGS.batch_size
    e_site = min(s_site+FLAGS.batch_size, x_len)
    # print('e_site %d' % e_site)
    x_y = data[s_site:e_site]
    x_batch = np.array(x_y[0], dtype=np.float32)
    y_batch = np.array(x_y[1], dtype=np.float32)
    lengths = np.array([np.shape(x_batch)[1]]*np.shape(x_batch)[0])
    if coord.should_stop():
      break
    loss = sess.run(val_model._loss,
                    feed_dict={val_model.inputs: x_batch,
                               val_model.labels: y_batch,
                               val_model.lengths: lengths})
    # print('validation %d loss %lf' % (i+1, loss/(e_site-s_site)))
    val_loss += loss
  val_loss /= x_len

  return val_loss


def train():

  # data_dir = '/home/student/work/pit_test/data_small'
  data_dir = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set'
  data_mixed = mixed_aishell.read_data_sets(data_dir)

  g = tf.Graph()
  with g.as_default():
    with tf.name_scope('model'):
      # with tf.variable_scope('lstm_var',reuse=tf.AUTO_REUSE):
      # tr_model and val_model should share variables
      tr_model = LSTM(FLAGS)
      tf.get_variable_scope().reuse_variables()
      val_model = LSTM(FLAGS)
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    # Prevent exhausting all the gpu memories.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.allow_soft_placement = True
    # sess = tf.InteractiveSession(config=config)
    sess = tf.Session(config=config)
    sess.run(init)
    if FLAGS.resume_training.lower() == 'true':
      ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir + '/nnet')
      if ckpt and ckpt.model_checkpoint_path:
        tf.logging.info("restore from" + ckpt.model_checkpoint_path)
        tr_model.saver.restore(sess, ckpt.model_checkpoint_path)
        best_path = ckpt.model_checkpoint_path
      else:
        tf.logging.fatal("checkpoint not found")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # g.finalize()

    # train_X_Y = data_mixed.train.X_Y
    # lengths = np.array([np.shape(train_X_Y[:10][0])[1]]
    #                    * np.shape(train_X_Y[:10][0])[0])
    # print(sess.run(val_model.costshape,
    #                feed_dict={val_model.inputs: train_X_Y[:10][0],
    #                           val_model.labels: train_X_Y[:10][1],
    #                           val_model.lengths: lengths}))
    # exit(0)
    try:
      # Cross validation before training.
      loss_prev = eval_one_epoch(
          sess, coord, val_model, data_mixed.test_cc.X_Y)
      tf.logging.info("CROSSVAL PRERUN AVG.LOSS %.4F" % loss_prev)

      sess.run(tf.assign(tr_model.lr, FLAGS.learning_rate))
      for epoch in range(FLAGS.max_epochs):
        start_time = time.time()

        # Training
        train_X_Y = data_mixed.train.X_Y
        tr_loss = train_one_epoch(sess,
                                  coord,
                                  tr_model,
                                  train_X_Y)

        # Validation
        val_X_Y = data_mixed.test_cc.X_Y
        val_loss = eval_one_epoch(sess,
                                  coord,
                                  val_model,
                                  val_X_Y)

        end_time = time.time()
        # Determine checkpoint path
        ckpt_name = "nnet_iter%d_lrate%e_trloss%.4f_cvloss%.4f" % (
            epoch + 1, FLAGS.learning_rate, tr_loss, val_loss)
        ckpt_dir = FLAGS.save_dir + '/nnet'
        if not os.path.exists(ckpt_dir):
          os.makedirs(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        # Relative loss between previous and current val_loss
        rel_impr = (loss_prev - val_loss) / loss_prev
        # Accept or reject new parameters
        if val_loss < loss_prev:
          tr_model.saver.save(sess, ckpt_path)
          # Logging train loss along with validation loss
          loss_prev = val_loss
          best_path = ckpt_path
          tf.logging.info(
              "ITERATION %d: TRAIN AVG.LOSS %.4f, (lrate%e) CROSSVAL"
              " AVG.LOSS %.4f, %s (%s), TIME USED: %.2fs" % (
                  epoch + 1, tr_loss, FLAGS.learning_rate, val_loss,
                  "nnet accepted", ckpt_name,
                  (end_time - start_time) / 1))
        else:
          tr_model.saver.restore(sess, best_path)
          tf.logging.info(
              "ITERATION %d: TRAIN AVG.LOSS %.4f, (lrate%e) CROSSVAL"
              " AVG.LOSS %.4f, %s, (%s), TIME USED: %.2fs" % (
                  epoch + 1, tr_loss, FLAGS.learning_rate, val_loss,
                  "nnet rejected", ckpt_name,
                  (end_time - start_time) / 1))

        # Start halving when improvement is low
        if rel_impr < FLAGS.start_halving_impr:
          FLAGS.learning_rate *= FLAGS.halving_factor
          sess.run(tf.assign(tr_model.lr, FLAGS.learning_rate))

        # Stopping criterion
        if rel_impr < FLAGS.end_halving_impr:
          if epoch < FLAGS.min_epochs:
            tf.logging.info(
                "we were supposed to finish, but we continue as "
                "min_epochs : %s" % FLAGS.min_epochs)
            continue
          else:
            tf.logging.info(
                "finished, too small rel. improvement %g" % rel_impr)
            break
    except Exception as e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      # Wait for threads to finish.
      coord.join(threads)

    tf.logging.info("Done training")
    sess.close()


def main(_):
  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)
  if FLAGS.decode:
    decode()
  else:
    train()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--decode',
      type=int,
      default=0,
      # action='store_true',
      help="Flag indicating decoding or training."
  )
  parser.add_argument(
      '--decode_show_spec',
      type=int,
      default=1,
      help="Flag indicating show spectrum or not."
  )
  parser.add_argument(
      '--resume_training',
      type=str,
      default='False',
      help="Flag indicating whether to resume training from cptk."
  )
  # parser.add_argument(
  #     '--lists_dir',
  #     type=str,
  #     default='list',
  #     help="Directory to load train, val and test data."
  # )
  parser.add_argument(
      '--input_size',
      type=int,
      default=257,
      help="The dimension of input."
  )
  parser.add_argument(
      '--output_size',
      type=int,
      default=257,
      help="The dimension of output."
  )
  # parser.add_argument(
  #     '--czt_dim',
  #     type=int,
  #     default=0,
  #     help="chrip-z transform feats dimension. it should be 0 if you just use fft spectrum feats"
  # )

  parser.add_argument(
      '--rnn_size',
      type=int,
      default=496,
      help="Number of rnn units to use."
  )
  parser.add_argument(
      '--rnn_num_layers',
      type=int,
      default=2,
      help="Number of layer of rnn model."
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help="Mini-batch size."
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help="Initial learning rate."
  )
  parser.add_argument(
      '--min_epochs',
      type=int,
      default=10,
      help="Min number of epochs to run trainer without halving."
  )
  parser.add_argument(
      '--max_epochs',
      type=int,
      default=50,
      help="Max number of epochs to run trainer totally."
  )
  parser.add_argument(
      '--halving_factor',
      type=float,
      default=0.5,
      help="Factor for halving."
  )
  parser.add_argument(
      '--start_halving_impr',
      type=float,
      default=0.003,
      help="Halving when ralative loss is lower than start_halving_impr."
  )
  parser.add_argument(
      '--end_halving_impr',
      type=float,
      default=0.001,
      help="Stop when relative loss is lower than end_halving_impr."
  )
  # parser.add_argument(
  #     '--num_threads',
  #     type=int,
  #     default=12,
  #     help='The num of threads to read tfrecords files.'
  # )
  parser.add_argument(
      '--save_dir',
      type=str,
      default='exp/lstm_pit',
      help="Directory to put the train result."
  )
  parser.add_argument(
      '--keep_prob',
      type=float,
      default=0.8,
      help="Keep probability for training dropout."
  )
  parser.add_argument(
      '--max_grad_norm',
      type=float,
      default=5.0,
      help="The max gradient normalization."
  )
  parser.add_argument(
      '--assign',
      type=str,
      default='def',
      help="Assignment method, def or opt"
  )
  parser.add_argument(
      '--model_type',
      type=str, default='LSTM',
      help="BLSTM or LSTM"
  )
  FLAGS, unparsed = parser.parse_known_args()
  # pp.pprint(FLAGS.__dict__)
  sys.stdout.flush()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
