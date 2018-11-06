from abc import ABCMeta, abstractmethod
import tensorflow as tf
from dataManager.dataManager import BasicManager


class FastBasicModel(object):
  __metaclass__ = ABCMeta

  def __init__(self):
    self.session = tf.Session()
    self.x_ph = 1
    self.y_ph = 2
    self.predict_node_name = 3
    self.loss_node_name = 4
    self.train_node_name = 5
    self.graph_node_dict = self.__build_graph()
    self.session.run(tf.global_variables_initializer())

  @abstractmethod
  def __build_graph(self):
    '''
    return predict,loss,train_operation
    '''
    pass

  def train(self, data, batch_size, epoch):
    # data=BasicManager("","").load(data)
    for i_epoch in range(epoch):
      avg_lost = 0.0
      x_len = data.size()
      total_batch = (x_len//batch_size)+1
      for i in range(total_batch):
        s_site = i*batch_size
        e_site = min(s_site+batch_size, x_len)
        x_batch, y_batch = data.x_y_segment(s_site, e_site)
        loss_t, _ = self.session.run((self.graph_node_dict[self.loss_node_name],
                                      self.graph_node_dict[self.train_node_name]),
                                     feed_dict={self.graph_node_dict[self.x_ph]: x_batch,
                                                self.graph_node_dict[self.y_ph]: y_batch})
        avg_lost += float(loss_t)/total_batch
      log("NNET Training : Epoch"+' %04d' %
          (epoch+1)+" Lost "+str(avg_lost))
    log(: over)

  def predict(self, x):
    return self.session.run(self.graph_node_dict[self.predict_node_name],
                            feed_dict={self.x_ph: x})
