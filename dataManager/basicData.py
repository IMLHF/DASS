from abc import ABCMeta, abstractmethod
import copy
import shutil
import os
import traceback
import scipy.io
from logger.log import LOGGER

LOG_ROOT = '_log/dataManager'


class IndexableData(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def __len__(self):
    pass

  @abstractmethod
  def __raw_getitem__(self, begin, end):
    pass

  def __getitem_by_slice__(self, index):
    start = 0 if index.start is None else index.start
    stop = self.__len__() if index.stop is None else index.stop
    step = 1 if index.step is None else index.step
    if step != 1:
      traceback.print_stack()
      raise Exception('Data not support jump!')
    if start < 0:
      start += self.__len__()
    if stop < 0:
      stop += self.__len__()
    if start < 0 or stop < 0 or start >= self.__len__() or stop > self.__len__():
      traceback.print_stack()
      raise Exception('Index out of range!')
    if start >= stop:
      return []
    return self.__raw_getitem__(start, stop)

  def __getitem_by_int__(self, index):
    if index < 0:
      index += self.__len__()
    if index < 0 or index >= self.__len__():
      traceback.print_stack()
      raise Exception('Index out of range!')
    return self.__raw_getitem__(index, index+1)

  def __getitem_by_str__(self, index):
    traceback.print_stack()
    raise Exception('Data not support index by str!')

  def __getitem_by_tuple__(self, index):
    traceback.print_stack()
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
      traceback.print_stack()
      raise Exception('Data not support !')


class Data(IndexableData):

  def __init__(self, data_index_list, itemgetor_by_data_index):
    '''
    type(data_index_list) is list.
    data_index_list : list of data index.
    '''
    self.data_index_list = data_index_list
    self.itemgetor_by_data_index = itemgetor_by_data_index

  def __raw_getitem__(self, begin, end):
    data = []
    for data_index in self.data_index_list[begin:end]:
      data.append(self.itemgetor_by_data_index(data_index))
    return data

  def __len__(self):
    return len(self.data_index_list)


class SET:
  def __init__(self, data_index_list, itemgetor_list):
    self.X = Data(data_index_list, itemgetor_list[0])
    self.Y = Data(data_index_list, itemgetor_list[1])
    self.X_Y = Data(data_index_list, itemgetor_list[2])

# region API
logger=copy.deepcopy(LOGGER(LOG_ROOT))
class DATABASE:
  def __init__(self, set_dict, itemgetor_list):
    set_name_list = set_dict.keys()
    for set_name in set_name_list:
      self.__setattr__(set_name, SET(set_dict[set_name], itemgetor_list))
# endregion
