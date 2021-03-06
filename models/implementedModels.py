from models.basicModel import SimpleBasicModel
import tensorflow as tf
import numpy as np
from utils.tf_tool import sigmoid_tdnn_layer
from utils import tf_tool


class DEEP_SPEECH_SEPARTION(SimpleBasicModel):
  """
  description
  ----------
  speech separation, implement using CNN.
  alternative to use losses.loss.PIT_MSE/losses.loss.MSE as loss function.
  ----------
  Parameters
  ----------
  layers_size : ``list`` or ``np.ndarray``
    the number of units of CNN layers.
  time_width : ``list`` or ``np.ndarray``
    delay time of CNN layers.
  loss_fun : ``losses.loss.*`` function
    loss function, please use function in package {losses.loss} .
  learning_rate : ``float``
    learning rate.
  gpu_list : ``list``
    list of gpu to use, e.g.[0,1,3] to use '/gpu:0', '/gpu:1' and '/gpu:3'
  name : ``str``
    model name. log will be built by the name, don't include special numeric.
  """

  def __init__(self, layers_size, times_width, loss_fun, learning_rate, gpu_list, name):

    self.gpu_list = gpu_list
    self.loss_function = loss_fun
    self.learning_rat = learning_rate
    self.layers_size = layers_size
    self.times_width = times_width
    self.debug = []  # TODO rm
    SimpleBasicModel.__init__(self, name)

  def _build_graph(self):
    with tf.Graph().as_default():
      optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rat)
      x_in = tf.placeholder(
          "float32", shape=[None, None, self.layers_size[0]])
      y_reference = tf.placeholder(
          "float32", shape=[None, None, self.layers_size[-1]])
      # print(x_in.name)
      # print(y_reference.name)

      n_gpu = len(self.gpu_list)
      n_x = tf.shape(x_in)[0]
      # self.debug.append(tf.shape(x_in))  # TODO rm
      dataslice = tf_tool.get_gpu_batch_size_list(n_x, n_gpu)
      x_in_list = tf.split(x_in, dataslice, axis=0)
      y_reference_list = tf.split(y_reference, dataslice, axis=0)
      with tf.variable_scope('gpu_variables', reuse=tf.AUTO_REUSE):
        for i_gpu, gpu_id in enumerate(self.gpu_list):
          with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('tower_%d' % gpu_id):
              x_in_gpu_batch = x_in_list[i_gpu]
              y_refer_gpu_batch = y_reference_list[i_gpu]
              y_out = x_in_gpu_batch
              for i, units_num in enumerate(self.layers_size[1:]):
                y_out = sigmoid_tdnn_layer(
                    y_out, self.times_width[i], 1, units_num, 'SAME', 'tdnn_layer_' + str(i+1))

              y_out = tf.multiply(
                  tf.concat([x_in_gpu_batch, x_in_gpu_batch], axis=2), y_out)  # MASK

              loss = self.loss_function(y_refer_gpu_batch, y_out)
              grads = optimizer.compute_gradients(loss)
              tf_tool.tower_to_collection(tower_losses=loss,
                                          tower_grads=grads,
                                          tower_y_outs=y_out)

      aver_loss = tf.reduce_mean(
          tf.get_collection('tower_losses'), name='avg_loss')
      train_op = optimizer.apply_gradients(
          tf_tool.average_gradients(tf.get_collection('tower_grads')), name='train_op')
      tmpdict= {
          self.x_ph_nodeid: x_in,
          self.y_ph_nodeid: y_reference,
          self.predict_nid: tf.concat(tf.get_collection('tower_y_outs'), 0, name='predict'),
          self.loss_nid: aver_loss,
          self.train_nid: train_op,
      }
      # print(tmpdict[self.predict_nid].name)
      # exit(0)
      return tmpdict

