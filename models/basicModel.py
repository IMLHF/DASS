from abc import ABCMeta, abstractmethod, abstractproperty
import tensorflow as tf
from logger.log import LOGGER
import time
import datetime
import numpy as np


class SimpleBasicModel(object):
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
    # endregions
    self.graph_node_dict = self._build_graph()
    self.session.run(tf.global_variables_initializer())
    logger = self.get_logger()
    logger.rmdir(name)

  def __del__(self):
    self.session.close()

  @abstractmethod
  def _build_graph(self):
    '''
    return graph_node_dict
    '''
    raise NotImplementedError

  @abstractmethod
  def save_model(self):
    raise NotImplementedError

  def train(self, data, batch_size, epoch, verbose=True):
    logger = self.get_logger()
    logger.set_file(self.name+'/train.log')
    loggerbatch = self.get_logger()
    loggerbatch.set_file(self.name+'/uttnum.log')
    logger.print_save('Training...\nTraining start time: ' +
                      str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    train_start_time = time.time()
    for i_epoch in range(epoch):
      start_time = time.time()
      avg_lost = 0.0
      x_len = len(data)
      total_batch = x_len//batch_size if (x_len % batch_size == 0) else ((
          x_len//batch_size)+1)
      batch100_begin_time = time.time()
      for i in range(total_batch):
        s_site = i*batch_size
        e_site = min(s_site+batch_size, x_len)
        x_y = data[s_site:e_site]
        x_batch = np.array(x_y[0], dtype=np.float32)
        y_batch = np.array(x_y[1], dtype=np.float32)
        loss_t, _ = self.session.run((self.graph_node_dict[self.loss_nid],
                                      self.graph_node_dict[self.train_nid]),
                                     feed_dict={self.graph_node_dict[self.x_ph_nodeid]: x_batch,
                                                self.graph_node_dict[self.y_ph_nodeid]: y_batch})
        avg_lost += float(loss_t)
        tr_loss = avg_lost / (i*batch_size+e_site-s_site)
        if verbose and (i % 100 == 0):
          batch_cost_time = time.time()-batch100_begin_time
          batch100_begin_time=time.time()
          loggerbatch.print_save(self.name+" Training : Epoch"+' %04d' %
                                 (i_epoch+1)+", batch %04d. Average Loss: " % (i+1) + '%02.9lf' % tr_loss+' Cost time : ' + '%02.2lf' % batch_cost_time+'S')
      duration = time.time() - start_time
      avg_lost /= x_len
      if verbose:
        logger.print_save(self.name+" Training : Epoch"+' %04d' %
                          (i_epoch+1)+" Lost "+'%02.9lf' % avg_lost+' Cost time : ' + '%02.2lf' % duration+'S')
      else:
        logger.save(self.name+" Training : Epoch"+' %04d' %
                    (i_epoch+1)+" Lost "+'%02.9lf' % avg_lost+' Cost time : ' + '%02.2lf' % duration+'S')
    train_cost_time = time.time() - train_start_time
    if verbose:
      logger.print_save(
          "\n"+self.name+' Optimizer Finished. Cost time : ' + '%02.2lf' % train_cost_time+'S')
    else:
      logger.save("\n"+self.name +
                  ' Optimizer Finished. Cost time : ' + '%02.2lf' % train_cost_time+'S')
    logger.print_save(
        'Trainning complete time: '+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

  def test_MSE(self, data, batch_size):
    logger = self.get_logger()
    logger.set_file(self.name+'/test.log')
    x_len = len(data)
    total_batch = x_len//batch_size if (x_len % batch_size == 0) else ((
        x_len//batch_size)+1)
    mse_list = []
    for i in range(total_batch):
      s_site = i*batch_size
      e_site = min(s_site+batch_size, x_len)
      x_y = data[s_site:e_site]
      x = x_y[0]
      y = x_y[1]
      y_out = self.predict(x)
      mse = np.mean((y-y_out)**2)
      logger.print_save('Batch %04d MSE : %lf' % (i, mse))
      mse_list.append(mse)
    logger.print_save('Average Test MSE : %lf' % np.mean(mse_list))

  def predict(self, x):
    # logger = self.get_logger()
    # logger.set_file(self.name+'/predict.log')
    return self.session.run(self.graph_node_dict[self.predict_nid],
                            feed_dict={self.graph_node_dict[self.x_ph_nodeid]: x})

  @staticmethod
  def get_logger():
    LOG_ROOT = '_log/models'
    return LOGGER(LOG_ROOT)
