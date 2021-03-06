"""
    Build the LSTM(BLSTM)  neural networks for PIT speech separation.
    for run_lstm_pit_v2.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn
import numpy as np
from losses.loss import utt_PIT_MSE_for_LSTM
from FLAGS import NNET_PARAM as config


class LSTM(object):
  """Build BLSTM or LSTM model with PIT loss functions.

  Attributes:
      config: Used to config our model
              config.input_size: feature (input) size;
              config.output_size: the final layer(output layer) size;
              config.rnn_size: the rnn cells' number
              config.batch_size: the batch_size for training
              config.rnn_num_layers: the rnn layers numbers
              config.keep_prob: the dropout rate
      inputs_batch: the mixed speech feature with cmvn as the inputs of model(LSTM or BLSTM)
      labels1_batch: the spk1's feature, as targets to train the model
      labels2_batch: the spk2's feature, as targets to train the model
      lengths_batch: rnn sequence_lengths
      infer: bool, if training(false) or test (true)
  """

  def __init__(self, inputs_batch, label1_batch, label2_batch, lengths_batch, infer=False):
    self._inputs = inputs_batch
    self._mixed = self._inputs

    self._labels1 = label1_batch
    self._labels2 = label2_batch

    self._lengths = lengths_batch

    self.batch_size = tf.shape(self._inputs)[0]

    self._model_type = config.model_type

    self.current_batchsize = tf.shape(self._lengths)[0]

    outputs = self._inputs
    # This first layer-- feed forward layer
    # Transform the input to the right size before feed into RNN

    with tf.variable_scope('forward1'):
      outputs = tf.reshape(outputs, [-1, config.input_size])
      outputs = tf.layers.dense(outputs, units=config.rnn_size,
                                activation=tf.nn.tanh,
                                reuse=tf.get_variable_scope().reuse)
      # print(outputs.name,'__________________________________________________')
      # print([x.name for x in tf.global_variables()],'_______________________________________')
      outputs = tf.reshape(
          outputs, [self.batch_size, -1, config.rnn_size])

    def lstm_cell():
      return tf.contrib.rnn.LSTMCell(
          config.rnn_size, forget_bias=1.0, use_peepholes=True,
          initializer=tf.contrib.layers.xavier_initializer(),
          state_is_tuple=True, activation=tf.tanh)
    attn_cell = lstm_cell
    if not infer and config.keep_prob < 1.0:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)

    if config.model_type.lower() == 'blstm':
      with tf.variable_scope('blstm'):

        lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.rnn_num_layers)], state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.rnn_num_layers)], state_is_tuple=True)

        lstm_fw_cell = _unpack_cell(lstm_fw_cell)
        lstm_bw_cell = _unpack_cell(lstm_bw_cell)
        result = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=lstm_fw_cell,
            cells_bw=lstm_bw_cell,
            inputs=outputs,
            dtype=tf.float32,
            sequence_length=self._lengths)
        outputs, fw_final_states, bw_final_states = result
    if config.model_type.lower() == 'lstm':
      with tf.variable_scope('lstm'):
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.rnn_num_layers)],
            state_is_tuple=True)
        self._initial_state = cell.zero_state(self.batch_size, tf.float32)
        state = self.initial_state
        outputs, state = tf.nn.dynamic_rnn(
            cell, outputs,
            dtype=tf.float32,
            sequence_length=tf.cast(self._lengths,tf.int32),
            initial_state=self.initial_state)
        self._final_state = state

    with tf.variable_scope('forward2'):
      if self._model_type.lower() == 'blstm':
        outputs = tf.reshape(outputs, [-1, 2*config.rnn_size])
        in_size = 2*config.rnn_size
      else:
        outputs = tf.reshape(outputs, [-1, config.rnn_size])
        in_size = config.rnn_size
      out_size = config.output_size
      # in_size=256
      weights1 = tf.get_variable('weights1', [in_size, out_size],
                                 initializer=tf.random_normal_initializer(stddev=0.01))
      biases1 = tf.get_variable('biases1', [out_size],
                                initializer=tf.constant_initializer(0.0))
      weights2 = tf.get_variable('weights2', [in_size, out_size],
                                 initializer=tf.random_normal_initializer(stddev=0.01))
      biases2 = tf.get_variable('biases2', [out_size],
                                initializer=tf.constant_initializer(0.0))
      mask1 = tf.nn.relu(tf.matmul(outputs, weights1) + biases1)
      mask2 = tf.nn.relu(tf.matmul(outputs, weights2) + biases2)
      self._activations1 = tf.reshape(
          mask1, [self.batch_size, -1, config.output_size])
      self._activations2 = tf.reshape(
          mask2, [self.batch_size, -1, config.output_size])

      self._cleaned1 = self._activations1*self._mixed
      self._cleaned2 = self._activations2*self._mixed
    # Ability to save the model
    self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)
    if infer:
      return

    self._loss = utt_PIT_MSE_for_LSTM(self._cleaned1,
                                      self._cleaned2,
                                      self._labels1,
                                      self._labels2)
    if tf.get_variable_scope().reuse:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(self.lr)
    #optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name='new_learning_rate')
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def cleaned1(self):
    return self._cleaned1

  @property
  def cleaned2(self):
    return self._cleaned2

  @property
  def inputs(self):
    return self._inputs

  @property
  def labels(self):
    return self._labels1, self._labels2

  @property
  def lengths(self):
    return self._lengths

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def activations(self):
    return self._activations

  @property
  def loss(self):
    return self._loss

  @property
  def train_op(self):
    return self._train_op

  def get_opt_output(self):
    '''
        This function is just for the PIT testing with optimal assignment
    '''

    cost1 = tf.reduce_sum(tf.pow(self._cleaned1-self._labels1, 2), 2) + \
        tf.reduce_sum(tf.pow(self._cleaned2-self._labels2, 2), 2)
    cost2 = tf.reduce_sum(tf.pow(self._cleaned2-self._labels1, 2), 2) + \
        tf.reduce_sum(tf.pow(self._cleaned1-self._labels2, 2), 2)
    idx = tf.slice(cost1, [0, 0], [1, -1]) > tf.slice(cost2, [0, 0], [1, -1])
    idx = tf.cast(idx, tf.float32)
    idx = tf.reduce_mean(idx, reduction_indices=0)
    idx = tf.reshape(idx, [tf.shape(idx)[0], 1])
    x1 = self._cleaned1[0, :, :] * (1-idx) + self._cleaned2[0, :, :]*idx
    x2 = self._cleaned1[0, :, :]*idx + self._cleaned2[0, :, :]*(1-idx)
    row = tf.shape(x1)[0]
    col = tf.shape(x1)[1]
    x1 = tf.reshape(x1, [1, row, col])
    x2 = tf.reshape(x2, [1, row, col])
    return x1, x2


def _unpack_cell(cell):
  if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
    return cell._cells
  else:
    return [cell]
