from models import basicModel
import tensorflow as tf
import numpy as np


class DBN(object):
  def __init__(self, sizes, learning_rate=0.01, cd_k=1):
    self._sizes = sizes
    self._rbm_list = []
    for i, size in enumerate(self._sizes[1:]):
      visible_size = self._sizes[i]
      hidden_size = size
      self._rbm_list.append(RBM(
          "RBM_%d" % i, visible_size, hidden_size, learning_rate=learning_rate, CDk=cd_k))

  def pretrain(self, x, batch_size, n_epoches):
    total_batch=len(x)//
    for rbm in self._rbm_list:
      for i in total_batch
      rbm.rbm_train(x, batch_size, n_epoches)
      x = rbm.rbm_forward(x)


class RBM(basicModel.FastBasicModel):
  def __init__(self, name, isize, osize, learning_rate, CDk=1):
    basicModel.FastBasicModel.__init__(self,name)
    self._input_size = isize
    self._output_size = osize
    self._learning_rate = learning_rate
    self._CDk = CDk

  def __build_graph(self):
    # with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
    self._weight = tf.get_variable(self.name+"_weight", [self._input_size, self._output_size],
                                   initializer=tf.random_normal_initializer())
    self._v_bias = tf.get_variable(self.name+"_v_bias", [self._input_size],
                                   initializer=tf.random_normal_initializer())
    self._h_bias = tf.get_variable(self.name+"_h_bias", [self._output_size],
                                   initializer=tf.random_normal_initializer())

    x_in = tf.placeholder(tf.float32, shape=[None, self._input_size])
    rbm_pretrain = self._rbm_train_epoche(x_in)
    x_loss = self.reconstruct(x_in)

    node_dict = {
      self.x_ph_nodeid:x_in,
      self.train_nid:rbm_pretrain,
      self.loss_nid:x_loss,
    }
    return node_dict

  def _0_1_sample_given_p(self, p):
    return tf.nn.relu(tf.sign(p - tf.random_uniform(tf.shape(p))))

  def _predict_h_given_v(self, v):
    prob = tf.nn.sigmoid(
        tf.matmul(v, self._weight)+self._h_bias)
    return prob, self._0_1_sample_given_p(prob)

  def _predict_v_given_h(self, h):
    prob = tf.nn.sigmoid(
        tf.matmul(h, tf.transpose(self._weight))+self._v_bias)
    return prob, self._0_1_sample_given_p(prob)

  def _CDk_f(self, vis):
    v0_prob = vis
    h0_prob, h0_sample = self._predict_h_given_v(v0_prob)
    hk_sample = h0_sample
    hk_prob = h0_sample
    for i in range(self._CDk):
      vk_prob, vk_sample = self._predict_v_given_h(hk_prob)  # 隐层使用概率
      # vk_prob, vk_sample = self._predict_v_given_h(hk_sample)  # 隐层使用逻辑单元
      hk_prob, hk_sample = self._predict_h_given_v(vk_prob)   # 可视层使用概率代替

    delta_w_positive = tf.matmul(tf.transpose(v0_prob), h0_prob)
    delta_w_negative = tf.matmul(tf.transpose(vk_prob), hk_prob)

    delta_w = tf.subtract(delta_w_positive, delta_w_negative) / \
        tf.to_float(tf.shape(v0_prob)[0])
    delta_vb = tf.reduce_mean(v0_prob-vk_prob, 0)
    delta_hb = tf.reduce_mean(h0_prob-hk_prob, 0)

    return delta_w, delta_vb, delta_hb

  def _rbm_train_epoche(self, vis):
    delta_w, delta_vb, delta_hb = self._CDk_f(vis)
    # update rbm parameters
    update_w_op = self._weight.assign_add(self._learning_rate*delta_w)
    update_vb_op = self._v_bias.assign_add(self._learning_rate*delta_vb)
    update_hb_op = self._h_bias.assign_add(self._learning_rate*delta_hb)

    return [update_w_op, update_vb_op, update_hb_op]

  def reconstruct(self, vis):
    _, h_samp = self._predict_h_given_v(vis)
    for i in range(self._CDk):
      v_recon, _ = self._predict_v_given_h(h_samp)
      _, h_samp = self._predict_h_given_v(v_recon)
    return tf.reduce_mean(tf.square(vis - v_recon))

  def rbm_forward(self, vis):
    assert np.shape(vis)[1] == self._input_size
    x_up, _ = self._predict_h_given_v(vis)
    return self.session.run(x_up)

  def get_param(self):
    return self.session.run(self._weight), self.session.run(self._v_bias), self.session.run(self._h_bias)
