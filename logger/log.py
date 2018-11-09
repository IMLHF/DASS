import traceback
import os
import shutil


class LOGGER(object):
  def __init__(self, root):
    self.root = root
    self.file = None

  def rmdir(self,file):
    file=self.root+'/'+file
    if os.path.exists(file):
      shutil.rmtree(file)
    os.makedirs(file)

  def set_file(self, file=""):
    self.file = file
    dir_=self.file_dir()
    dir_=dir_[:dir_.rfind('/')]
    if not os.path.exists(dir_):
      os.makedirs(dir_)

  def file_dir(self):
    return self.root+'/'+self.file

  def print_save(self, msg):
    print(msg)
    if self.file is None:
      traceback.print_stack()
      raise Exception(
          'please use logger.set_file() before print or save massage!')
    with open(self.root+'/'+self.file, 'a+') as f:
      f.writelines(str(msg)+'\n')

  def save(self, msg):
    if self.file is None:
      traceback.print_stack()
      raise Exception('please use set!')
    with open(self.root+'/'+self.file, 'a+') as f:
      f.writelines(str(msg)+'\n')
