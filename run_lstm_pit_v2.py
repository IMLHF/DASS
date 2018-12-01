# run without TFRecord, more efficient.
import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf
from models.lstm_pit_v2 import LSTM
from dataManager.data import mixed_aishell
import utils
import wave
import shutil
import traceback
from dataManager.data import mixed_aishell_tfrecord_io
from FLAGS import NNET_PARAM
# from dataManager.data import mixed_aishell_tfrecord_io_repeat as mixed_aishell_tfrecord_io


os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]


def decode():
  pass
  """Decoding the inputs using current model."""
  '''
  # data_dir = '/home/student/work/pit_test/data_small'
  # data_dir = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set'
  data_dir = '/mnt/d/tf_recipe/ALL_DATA/aishell/mixed_data_small'
  data_mixed = mixed_aishell.read_data_sets(data_dir)

  with tf.Graph().as_default():
    with tf.name_scope('model'):
      model = LSTM(NNET_PARAM, infer=True)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    sess = tf.Session()

    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(NNET_PARAM.save_dir+'/nnet')
    if ckpt and ckpt.model_checkpoint_path:
      tf.logging.info("Restore from " + ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      tf.logging.fatal("checkpoint not found.")
      sys.exit(-1)

  speech_num = 10

  # speech_start = 100000  # same gender
  # speech_start = 100123 # differ gender
  # speech_start = 202810 # differ gender like norm
  # dataset=data_mixed.train
  # X_Y_batch = dataset.X_Y[speech_start:speech_start+speech_num]

  speech_start = 3128  # test_cc
  dataset = data_mixed.test_cc
  X_Y_batch = dataset.X_Y[speech_start:speech_start+speech_num]

  angle_batch = np.array(
      dataset.X_Theta[speech_start:speech_start+speech_num])
  x_batch = X_Y_batch[0]
  y_batch = X_Y_batch[1]
  lengths = np.array([np.shape(x_batch)[1]]*np.shape(x_batch)[0])
  cleaned1, cleaned2 = sess.run(
      [model.cleaned1, model.cleaned2],
      feed_dict={
          model.inputs: x_batch,
          model.labels: y_batch,
          model.lengths: lengths,
      })

  raw_spec1, raw_spec2 = np.split(y_batch, 2, axis=-1)

  cleaned1 = np.array(mixed_aishell.rmNormalization(cleaned1))
  cleaned2 = np.array(mixed_aishell.rmNormalization(cleaned2))
  raw_spec1 = np.array(mixed_aishell.rmNormalization(raw_spec1))
  raw_spec2 = np.array(mixed_aishell.rmNormalization(raw_spec2))
  mixed_spec2 = np.array(mixed_aishell.rmNormalization(x_batch))

  decode_ans_dir = os.path.join(NNET_PARAM.save_dir, 'decode_ans')
  if os.path.exists(decode_ans_dir):
    shutil.rmtree(decode_ans_dir)
  os.makedirs(decode_ans_dir)

  if NNET_PARAM.decode_show_spec:
    cleaned = np.concatenate([cleaned1, cleaned2], axis=-1)
    raw_spec = np.concatenate([raw_spec1, raw_spec2], axis=-1)
    utils.spectrum_tool.picture_spec(np.log10(cleaned+0.001),
                                     decode_ans_dir+'/restore_spec_')
    utils.spectrum_tool.picture_spec(np.log10(raw_spec+0.001),
                                     decode_ans_dir+'/raw_spec_')

  spec1 = cleaned1 * np.exp(angle_batch*1j)
  spec2 = cleaned2 * np.exp(angle_batch*1j)
  raw_spec1 = raw_spec1 * np.exp(angle_batch*1j)
  raw_spec2 = raw_spec2 * np.exp(angle_batch*1j)
  mixed_spec2 = mixed_spec2 * np.exp(angle_batch*1j)

  for i in range(speech_num):
    # write restore wave
    reY1 = utils.spectrum_tool.librosa_istft(
        spec1[i].T, (NNET_PARAM.input_size-1)*2, NNET_PARAM.input_size-1)
    reY2 = utils.spectrum_tool.librosa_istft(
        spec2[i].T, (NNET_PARAM.input_size-1)*2, NNET_PARAM.input_size-1)
    reCONY = np.concatenate([reY1, reY2])
    wavefile = wave.open(
        decode_ans_dir+('/restore_audio_%03d.wav' % i), 'wb')
    nchannels = 1
    sampwidth = 2  # 采样位宽，2表示16位
    framerate = 16000
    nframes = len(reCONY)
    comptype = "NONE"
    compname = "not compressed"
    wavefile.setparams((nchannels, sampwidth, framerate, nframes,
                        comptype, compname))
    wavefile.writeframes(
        np.array(reCONY, dtype=np.int16))

    # write raw wave
    rawY1 = utils.spectrum_tool.librosa_istft(
        raw_spec1[i].T, (NNET_PARAM.input_size-1)*2, NNET_PARAM.input_size-1)
    rawY2 = utils.spectrum_tool.librosa_istft(
        raw_spec2[i].T, (NNET_PARAM.input_size-1)*2, NNET_PARAM.input_size-1)
    rawCONY = np.concatenate([rawY1, rawY2])
    wavefile = wave.open(
        decode_ans_dir+('/raw_audio_%03d.wav' % i), 'wb')
    nframes = len(rawCONY)
    wavefile.setparams((nchannels, sampwidth, framerate, nframes,
                        comptype, compname))
    wavefile.writeframes(
        np.array(rawCONY, dtype=np.int16))

    # write mixed wave
    mixedWave = utils.spectrum_tool.librosa_istft(
        mixed_spec2[i].T, (NNET_PARAM.input_size-1)*2, NNET_PARAM.input_size-1)
    wavefile = wave.open(
        decode_ans_dir+('/mixed_audio_%03d.wav' % i), 'wb')
    nframes = len(mixedWave)
    wavefile.setparams((nchannels, sampwidth, framerate, nframes,
                        comptype, compname))
    wavefile.writeframes(
        np.array(mixedWave, dtype=np.int16))

    # wave picture
    utils.spectrum_tool.picture_wave(reCONY,
                                     decode_ans_dir +
                                     ('/restore_wav_%03d' % i),
                                     16000)
    utils.spectrum_tool.picture_wave(rawCONY,
                                     decode_ans_dir +
                                     ('/raw_wav_%03d' % i),
                                     16000)

  tf.logging.info("Done decoding.")
  sess.close()
  '''


def train_one_epoch(sess, tr_model):
  """Runs the model one epoch on given data."""
  tr_loss = 0
  i = 0
  while True:
    try:
      stime = time.time()
      print('getin')
      _, loss, current_batchsize = sess.run(
          [tr_model.train_op, tr_model.loss, tr_model.current_batchsize])
      tr_loss += loss
      # if (i+1) % int(100*256/NNET_PARAM.batch_size) == 0:
      lr = sess.run(tr_model.lr)
      costtime = time.time()-stime
      print("MINIBATCH %d: TRAIN AVG.LOSS %f, "
            "(learning rate %e)" % (
                i + 1, tr_loss / (i*NNET_PARAM.batch_size+current_batchsize), lr), 'cost time: %f' % costtime)
      sys.stdout.flush()
      i += 1
    except tf.errors.OutOfRangeError:
      break
  tr_loss /= ((i-1)*NNET_PARAM.batch_size+current_batchsize)
  return tr_loss


def eval_one_epoch(sess, val_model):
  """Cross validate the model on given data."""
  val_loss = 0
  data_len = 0
  while True:
    try:
      loss, current_batchsize = sess.run(
          [val_model.loss, val_model.current_batchsize])
      val_loss += loss
      data_len += current_batchsize
    except tf.errors.OutOfRangeError:
      break
  val_loss /= data_len
  return val_loss


def train():

  g = tf.Graph()
  with g.as_default():
    with tf.name_scope('model'):
      # region DataSet
      train_tfrecords, val_tfrecords, testcc_tfrecords = mixed_aishell_tfrecord_io.gen_tfrecord(
          # '/mnt/d/tf_recipe/ALL_DATA/aishell/mixed_data_small',
          '/home/student/work/pit_test/data',
          # 'feature_tfrecords_utt03s_test',
          '/big-data/tmplhf/pit-data/feature_tfrecords_utt10s',
          gen=True)
      exit(0)
      # train_epoch_data_size = 405600
      # val_epoch_data_size = 5400

      train_files = tf.data.Dataset.list_files(train_tfrecords)
      train_set = train_files.interleave(tf.data.TFRecordDataset,
                                         cycle_length=1)
      train_set = train_set.map(
          map_func=mixed_aishell_tfrecord_io.parse_func,
          num_parallel_calls=NNET_PARAM.num_threads_processing_data)
      train_set = train_set.padded_batch(
          NNET_PARAM.batch_size,
          padded_shapes=([None, NNET_PARAM.input_size],
                         [None, NNET_PARAM.output_size],
                         [None, NNET_PARAM.output_size],
                         []))
      train_set = train_set.prefetch(buffer_size=NNET_PARAM.batch_size)
      iter_train = train_set.make_initializable_iterator()
      x_batch_tr, y1_batch_tr, y2_batch_tr, lengths_batch_tr = iter_train.get_next()
      tr_model = LSTM(x_batch_tr, y1_batch_tr, y2_batch_tr, lengths_batch_tr)
      tf.get_variable_scope().reuse_variables()

      val_files = tf.data.Dataset.list_files(val_tfrecords)
      val_set = val_files.interleave(tf.data.TFRecordDataset,
                                     cycle_length=1)
      val_set = val_set.map(
          map_func=mixed_aishell_tfrecord_io.parse_func,
          num_parallel_calls=NNET_PARAM.num_threads_processing_data)
      val_set = val_set.padded_batch(
          NNET_PARAM.batch_size,
          padded_shapes=([None, NNET_PARAM.input_size],
                         [None, NNET_PARAM.output_size],
                         [None, NNET_PARAM.output_size],
                         []))
      # val_set = val_set.apply(tf.data.experimental.map_and_batch(
      #     map_func=mixed_aishell_tfrecord_io.parse_func,
      #     batch_size=NNET_PARAM.batch_size,
      #     num_parallel_calls=NNET_PARAM.num_threads_processing_data,
      #     # num_parallel_batches=NNET_PARAM.num_threads_processing_data
      # ))
      val_set = val_set.prefetch(buffer_size=NNET_PARAM.batch_size)
      iter_val = val_set.make_initializable_iterator()
      x_batch_val, y1_batch_val, y2_batch_val, lengths_batch_val = iter_val.get_next()
      val_model = LSTM(x_batch_val, y1_batch_val,
                       y2_batch_val, lengths_batch_val)
      # endregion

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)
    if NNET_PARAM.resume_training.lower() == 'true':
      ckpt = tf.train.get_checkpoint_state(NNET_PARAM.save_dir + '/nnet')
      if ckpt and ckpt.model_checkpoint_path:
        tf.logging.info("restore from" + ckpt.model_checkpoint_path)
        tr_model.saver.restore(sess, ckpt.model_checkpoint_path)
        best_path = ckpt.model_checkpoint_path
      else:
        tf.logging.fatal("checkpoint not found")
    # g.finalize()

    try:
      # validation before training.
      sess.run(iter_val.initializer)
      loss_prev = eval_one_epoch(sess,
                                 val_model)
      tf.logging.info("CROSSVAL PRERUN AVG.LOSS %.4F" % loss_prev)

      sess.run(tf.assign(tr_model.lr, NNET_PARAM.learning_rate))
      for epoch in range(NNET_PARAM.max_epochs):
        sess.run([iter_train.initializer, iter_val.initializer])
        start_time = time.time()

        # Training
        # print('shape')
        # print(sess.run([tf.shape(x_batch_tr), tf.shape(y1_batch_tr),
        #                 tf.shape(y2_batch_val), tf.shape(lengths_batch_tr)]))
        # print('time prepare data :', time.time()-start_time)
        tr_loss = train_one_epoch(sess,
                                  tr_model)
        # exit(0)

        # Validation
        val_loss = eval_one_epoch(sess,
                                  val_model)

        end_time = time.time()
        # Determine checkpoint path
        ckpt_name = "nnet_iter%d_lrate%e_trloss%.4f_cvloss%.4f" % (
            epoch + 1, NNET_PARAM.learning_rate, tr_loss, val_loss)
        ckpt_dir = NNET_PARAM.save_dir + '/nnet'
        if not os.path.exists(ckpt_dir):
          os.makedirs(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        # Relative loss between previous and current val_loss
        rel_impr = np.abs(loss_prev - val_loss) / loss_prev
        # Accept or reject new parameters
        if val_loss < loss_prev:
          tr_model.saver.save(sess, ckpt_path)
          # Logging train loss along with validation loss
          loss_prev = val_loss
          best_path = ckpt_path
          tf.logging.info(
              "ITERATION %03d: TRAIN AVG.LOSS %.4f, (lrate%e) CROSSVAL"
              " AVG.LOSS %.4f, %s (%s), TIME USED: %.2fs" % (
                  epoch + 1, tr_loss, NNET_PARAM.learning_rate, val_loss,
                  "nnet accepted", ckpt_name,
                  (end_time - start_time) / 1))
        else:
          tr_model.saver.restore(sess, best_path)
          tf.logging.info(
              "ITERATION %03d: TRAIN AVG.LOSS %.4f, (lrate%e) CROSSVAL"
              " AVG.LOSS %.4f, %s, (%s), TIME USED: %.2fs" % (
                  epoch + 1, tr_loss, NNET_PARAM.learning_rate, val_loss,
                  "nnet rejected", ckpt_name,
                  (end_time - start_time) / 1))

        # Start halving when improvement is low
        if rel_impr < NNET_PARAM.start_halving_impr:
          NNET_PARAM.learning_rate *= NNET_PARAM.halving_factor
          sess.run(tf.assign(tr_model.lr, NNET_PARAM.learning_rate))

        # Stopping criterion
        if rel_impr < NNET_PARAM.end_halving_impr:
          if epoch < NNET_PARAM.min_epochs:
            tf.logging.info(
                "we were supposed to finish, but we continue as "
                "min_epochs : %s" % NNET_PARAM.min_epochs)
            continue
          else:
            tf.logging.info(
                "finished, too small rel. improvement %g" % rel_impr)
            break
    except Exception as e:
      print(e)

    sess.close()
    tf.logging.info("Done training")


def main(_):
  if not os.path.exists(NNET_PARAM.save_dir):
    os.makedirs(NNET_PARAM.save_dir)
  if NNET_PARAM.decode:
    decode()
  else:
    train()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
