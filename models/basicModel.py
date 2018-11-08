from abc import ABCMeta, abstractmethod, abstractproperty
import tensorflow as tf
from logger.log import LOGGER
import time
import datetime
import numpy as np


class FastBasicModel(object):
  __metaclass__ = ABCMeta

  def __init__(self, name):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self.session = tf.Session(config=config)
    self.name = name
    # region node ID
    self.x_ph_nodeid = 1
    self.y_ph_nodeid = 2
    self.predict_nid = 3
    self.loss_nid = 4
    self.train_nid = 5
    # endregion
    self.graph_node_dict = self._build_graph()
    self.session.run(tf.global_variables_initializer())

  # def __del__(self):
  #   self.session.close()

  @abstractmethod
  def _build_graph(self):
    '''
    return graph_node_dict
    '''
    raise NotImplementedError

  def train(self, data, batch_size, epoch, verbose=True):
    logger = self.get_logger()
    logger.set_file(self.name+'/train.log')
    loggertmp=self.get_logger()
    loggertmp.set_file(self.name+'/uttnum.log')
    logger.print_save('Training...\nTraining start time: ' +
                      str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    train_start_time=time.time()
    for i_epoch in range(epoch):
      start_time = time.time()
      avg_lost = 0.0
      x_len = len(data)
      total_batch = x_len//batch_size if (x_len % batch_size == 0) else ((
          x_len//batch_size)+1)
      for i in range(total_batch):
        loggertmp.print_save(str(i))
        s_site = i*batch_size
        e_site = min(s_site+batch_size, x_len)
        # print(x_len,s_site,e_site)
        x_y = data[s_site:e_site]
        # print(np.shape(x_y))
        x_batch = x_y[0]
        y_batch = x_y[1]
        # print(np.shape(x_y[0]),np.shape(x_y[1]))
        loss_t, _ = self.session.run((self.graph_node_dict[self.loss_nid],
                                      self.graph_node_dict[self.train_nid]),
                                     feed_dict={self.graph_node_dict[self.x_ph_nodeid]: x_batch,
                                                self.graph_node_dict[self.y_ph_nodeid]: y_batch})
        avg_lost += float(loss_t)/total_batch
      duration = time.time() - start_time
      start_time = time.time()
      if verbose:
        logger.print_save(self.name+" Training : Epoch"+' %04d' %
                          (epoch+1)+" Lost "+str(avg_lost)+' Cost time : ' + str(duration))
      else:
        logger.save(self.name+" Training : Epoch"+' %04d' %
                    (epoch+1)+" Lost "+str(avg_lost)+' Cost time : ' + str(duration))
    train_cost_time=time.time() - train_start_time
    if verbose:
      logger.print_save("\n"+self.name+' Optimizer Finished. Cost time : ' + str(train_cost_time))
    else:
      logger.save("\n"+self.name+' Optimizer Finished. Cost time : ' + str(train_cost_time))
    logger.print_save(
        'Trainning complete time: '+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

  @abstractmethod
  def test(self, x, y):
    raise NotImplementedError

  def test_MSE(self, x, y):
    logger = self.get_logger()
    logger.set_file(self.name+'/test.log')

  def predict(self, x):
    # logger = self.get_logger()
    # logger.set_file(self.name+'/predict.log')
    return self.session.run(self.graph_node_dict[self.predict_nid],
                            feed_dict={self.x_ph_nodeid: x})

  @staticmethod
  def get_logger():
    LOG_ROOT = '_log/models'
    return LOGGER(LOG_ROOT)
