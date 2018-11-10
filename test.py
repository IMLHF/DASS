from dataManager.data import mixed_aishell
from models.implementedModels import DEEP_SPEECH_SEPARTION
import tensorflow as tf
import numpy as np
from losses import loss
from abc import abstractmethod, ABCMeta


class A(object):
  __metaclass__ = ABCMeta

  def __init__(self):
    self.namea = "aaa"

  @abstractmethod
  def fun(self):
    print("function a : %s" % self.namea)

  def __del__(self):
    print(self.namea)


class B(A):
  def __init__(self):
    # 这一行解决了问题
    A.__init__(self)
    self.nameb = "bbb"

  def fun(self):
    print("function b : %s" % self.nameb)


def data_manager_test():
  data_mixed = mixed_aishell.read_data_sets(
      '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set')
  print(data_mixed.train.X_Y[0:2])
  print(np.shape(data_mixed.train.X_Y[0:2]))
  print(np.shape(data_mixed.train.Y[512:512+128]))
  # x = tf.placeholder('float', shape=[None, 257, None])
  # print(__file__.rfind('/'))


def concat_test():
  v1 = tf.get_variable('v1', shape=[2, 12], initializer=tf.ones_initializer(
      dtype=tf.float32), dtype=tf.float32)
  v2 = tf.get_variable('v2', shape=[2, 12], initializer=tf.random_uniform_initializer(
      maxval=-1., minval=1., seed=0), dtype=tf.float32)
  # 向当前计算图中添加张量集合
  tf.add_to_collection('v', v1)
  tf.add_to_collection('v', v2)
  y1 = tf.constant([[[1, 1, 1, 0, 0, 0]],[[0,0,0,1,1,1]]])
  y2 = tf.constant([[[1, 1, 1, 0, 0, 0]],[[1,1,1,0,0,0]]])
  yy = tf.constant([[1,2,3],[4,5,6]])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.get_collection('v')))
    tf.Graph.clear_collection(tf.get_default_graph(),name='v')
    print(sess.run(tf.get_collection('v')))
    # a, b = tf.split(v1, 2, axis=-1)
    # print(np.shape(sess.run(a)))
    # print(sess.run(loss.PIT_MSE_for_CNN(y1, y2)))
    # print(sess.run(tf.reduce_mean(yy,[0,1])))


def run():
  data_dir = '/home/student/work/pit_test/data_small'
  # data_dir = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set'
  data_mixed = mixed_aishell.read_data_sets(data_dir)

  pit_model = DEEP_SPEECH_SEPARTION(layers_size=[257, 2048, 2048, 2048, 514],
                                    times_width=[7, 1, 1, 1],
                                    loss_fun=loss.PIT_MSE_for_CNN,
                                    learning_rate=0.01,
                                    gpu_list=[0],
                                    name='PIT')
  pit_model.train(data_mixed.train.X_Y, batch_size=128,epoch=6)
  pit_model.test_PIT(data_mixed.test_cc.X_Y,batch_size=128)
  # pit_model.save_model()

  # conv_model = DEEP_SPEECH_SEPARTION(layers_size=[257, 2048, 2048, 2048, 514],
  #                                    times_width=[7, 1, 1, 1],
  #                                    loss_fun=loss.MSE,
  #                                    learning_rate=0.01,
  #                                    gpu_list=[1],
  #                                    name='CONV')
  # conv_model.train(data_mixed.train.X_Y, batch_size=128,epoch=6)
  # conv_model.test_PIT(data_mixed.test_cc.X_Y,batch_size=128)
  # conv_model.save_model()



if __name__ == "__main__":
  # data_manager_test()
  # concat_test()
  run()
  # x=tf.random_normal([10,257,7])
  # y=tf.random_normal([7])
  # print(x/y)

