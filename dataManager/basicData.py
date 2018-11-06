from abc import ABCMeta, abstractmethod
import copy
import shutil
import os
import traceback
import scipy.io

LOG_ROOT='_log/dataManager'
DATA_ROOT='_data'
class IndexableData(object):
  __metaclass__ = ABCMeta

  def __init__(self, rawdata_dir, data_name,set_name):
    self._size = -1
    self._size = self.__len__()
    self.rawdata_dir = rawdata_dir
    self._data_name=data_name
    self._set_name=set_name

  @abstractmethod
  def __len__(self):
    pass

  # TODO
  @abstractmethod
  def shape(self):
    pass

  @abstractmethod
  def __raw_getitem__(self, begin, end):
    pass

  def __getitem_by_slice__(self, index):
    start = 0 if index.start is None else index.start
    stop = self.__len__() if index.stop is None else index.stop
    step = 1 if index.step is None else index.step
    if step != 1:
      traceback.print_exc()
      raise Exception('Data not support jump!')
    if start < 0:
      start += self.__len__()
    if stop < 0:
      stop += self.__len__()
    if start < 0 or stop < 0 or start >= self.__len__() or stop > self.__len__():
      traceback.print_exc()
      raise Exception('Index out of range!')
    if start >= stop:
      return []
    return self.__raw_getitem__(start, stop)

  def __getitem_by_int__(self, index):
    if index < 0:
      index += self.__len__()
    if index < 0 or index >= self.__len__():
      traceback.print_exc()
      raise Exception('Index out of range!')
    return self.__raw_getitem__(index, index+1)

  def __getitem_by_str__(self, index):
    traceback.print_exc()
    raise Exception('Data not support index by str!')

  def __getitem_by_tuple__(self, index):
    traceback.print_exc()
    raise Exception('Data not support index by tuple!')

  def __getitem__(self, index):
    if type(index) is slice:
      return self.__getitem_by_slice__(index)
    elif type(index) is int:
      return self.__getitem_by_int__(index)
    elif type(index) is str:
      return self.__getitem_by_str__(index)
    elif type(index) is tuple:
      return self.__getitem_by_tuple__(index)
    else:
      traceback.print_exc()
      raise Exception('Data not support !')

    pass

  def data_site(self):
    return self.rawdata_dir

  def data_name(self):
    return self._data_name

  def set_name(self):
    return self.set_name
