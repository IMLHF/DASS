import numpy as np
import copy

def x_tran_test():
  class T(object):
    def __init__(self):
      self.val=2

    def fun(self,x):
      return x*self.val

  class A(object):
    def __init__(self):
      self.val=1
      self.lis=[]

    def get(self):
      val=self.val
      for tmp in self.lis:
        val=tmp(val)
      return val

    def translate_by(self,fun):
      tmp=copy.deepcopy(self)
      tmp.lis.append(fun)
      return tmp

  a=A()
  a2=a.translate_by(T().fun)
  print(a.get())
  print(a2.get())


def numpy_test():
  lis=[np.array((1,2,3)),np.array((1,2,)),np.array((1,2,3,4))]
  arr=np.array(lis)
  print(np.shape(arr))
  print(arr)

if __name__=="__main__":
  # x_tran_test()
  numpy_test()
