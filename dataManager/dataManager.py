from abc import ABCMeta, abstractmethod
import copy


class basicManager(object):
  __metaclass__ = ABCMeta

  def __init__(self, rawdata):
    self.rawdata=rawdata
    self.__init_data()
    self.x_translate_list=[]
    self.y_translate_list=[]
    self.log_dir='log/dataManager'

  @abstractmethod
  def __init_data(self):
    pass

  @abstractmethod
  def __x_raw_segment(self):
    '''
    return rawdata[x][begin:end]
    '''
    pass

  @abstractmethod
  def __y_raw_segment(self):
    '''
    return rawdata[y][begin:end]
    '''
    pass

  @abstractmethod
  def __x_y_raw_segment(self):
    '''
    return rawdata[x][begin:end],rawdata[y][begin:end]
    '''
    pass

  # final,nerver overried
  def x_segment(self,begin,end):
    '''
    return translated_data[x][begin:end]
    '''
    data=self.__x_raw_segment(begin,end)
    for tran in self.x_translate_list:
      data=tran(data)
    return data

  # final,nerver overried
  def y_segment(self,begin,end):
    '''
    return translated_data[y][begin:end]
    '''
    data=self.__y_raw_segment(begin,end)
    for tran in self.y_translate_list:
      data=tran(data)
    return data

  # final,nerver overried
  def x_y_segment(self,begin,end):
    '''
    return translated_data[begin:end]
    '''
    x_data,y_data=self.__x_y_raw_segment(begin,end)
    for tran in self.x_translate_list:
      x_data=tran(x_data)
    for tran in self.y_translate_list:
      y_data=tran(y_data)
    return x_data,y_data

  # final,never overried
  def x_translate_by(self,fun):
    '''
    return rawdata(x_translate_list+=fun)
    '''
    newdata=copy.deepcopy(self)
    newdata.x_translate_list.append(fun)
    return newdata

  # final,never overried
  def y_translate_by(self,fun):
    '''
    return rawdata(y_translate_list+=fun)
    '''
    newdata=copy.deepcopy(self)
    newdata.y_translate_list.append(fun)
    return newdata

  @abstractmethod
  def size(self):
    pass
